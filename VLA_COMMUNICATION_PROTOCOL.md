# VLA ↔ Benchmark 통신 규약 (temporal_vla)

> VLA 모델과 벤치마크를 서로 다른 Docker 컨테이너로 분리하고, **통일 HTTP API**로 통신시키기 위한 프로토콜 명세.
> 모델/벤치마크 N×M 조합을 N+M 구현만으로 커버하기 위한 "Sub-key 라우팅" 설계.
>
> 참조: `scripts/utils/vla_client.py`, `src/processor/factory.py`, `scripts/serve_*.py`, `README.md`

---

## 1. 설계 철학

```
Env obs ─▶ ObsProcessor ─▶ VLAClient(HTTP) ─▶ Model Server
                                                   │
Env step ◀─ ActionProcessor ◀─ VLAClient ◀─────────┘
```

- **모델 서버**는 자신의 native format을 `action.*` sub-key 단위로 쪼개 반환한다.
- **벤치마크 측 Processor**는 자기에게 필요한 sub-key만 조립하여 환경 action으로 변환한다.
- **Observation**도 동일하게 `observation.*` dot-key 네임스페이스로 통일하여, 있는 키만 채워 보내고 서버가 필요한 키만 꺼내 쓴다.
- 결과적으로 모델/벤치 추가 시 기존 코드 수정이 거의 없다.

---

## 2. HTTP 엔드포인트

모든 모델 서버(`scripts/serve_*.py`)는 FastAPI + uvicorn으로 다음 3개 엔드포인트를 제공한다.

### 2.1 `GET /health`

서버 메타데이터 조회. 벤치마크 측이 startup 시 polling하여 서버가 준비될 때까지 대기.

**응답 예시**
```json
{
  "status": "ok",
  "model": "xvla",
  "action_type": "absolute",
  "action_keys": ["action.eef_pos", "action.eef_rot6d", "action.gripper"],
  "n_action_steps": 30
}
```

**필수 필드**
| 필드 | 타입 | 설명 |
|---|---|---|
| `status` | str | `"ok"` | `"loading"` 등 |
| `action_type` | str | `"relative"` 또는 `"absolute"` |
| `action_keys` | list[str] | 서버가 반환하는 action sub-key 목록 |
| `n_action_steps` | int | 1회 예측당 반환하는 action step 수 |

### 2.2 `POST /reset`

에피소드 히스토리 초기화 (history-based 모델용). Body는 비어도 됨.

### 2.3 `POST /act`

관측값 → 액션 예측.

**요청 body (JSON)**
```json
{
  "observation.images.static":  "<base64 PNG>",
  "observation.images.wrist":   "<base64 PNG>",
  "observation.state.eef_pos":  [x, y, z],
  "observation.state.eef_quat": [qx, qy, qz, qw],
  "observation.state.gripper_opening": [g],
  "task": "open the drawer"
}
```

**응답 body (JSON)** — sub-key 방식
```json
{
  "action.eef_pos":    [[x,y,z], ...],
  "action.eef_rot6d":  [[a,b,c,d,e,f], ...],
  "action.gripper":    [[g], ...],
  "latency_ms": 42.7
}
```

모든 action sub-key 값은 **항상 2D 리스트** `[n_steps, dim]`. 1D일 경우 클라이언트가 자동으로 `np.newaxis`를 추가한다.

---

## 3. Observation 스키마

Dot-notation을 그대로 JSON key로 사용한다. 벤치마크는 자신이 관측할 수 있는 키만 채워 보내며, 서버는 필요한 키만 꺼내 쓰되 없는 키는 자체 변환하거나 기본값으로 채운다.

### 3.1 이미지 — `observation.images.{camera}`
- 인코딩: **base64 PNG** (HWC uint8, RGB).
- 유틸: `scripts/utils/vla_client.py :: encode_image()` / `decode_image()`.
- 표준 카메라명:
  - `static` — 3인칭 고정 카메라 (Calvin `rgb_static`, RoboCasa `agentview_left_image`)
  - `wrist` — 손목 카메라 (Calvin `rgb_gripper`, RoboCasa `eye_in_hand_image`)
  - 그 외 임의 카메라명 허용 (`side_0`, `side_1` 등)

### 3.2 상태 — `observation.state.{field}`

모두 float list. 있는 키만 보내면 된다.

| 키 | shape | 설명 |
|---|---|---|
| `eef_pos` | `[3]` | End-effector 위치 (x, y, z) |
| `eef_quat` | `[4]` | End-effector 회전 쿼터니언 (xyzw 순) |
| `eef_euler` | `[3]` | End-effector 회전 Euler (Z-Y-X 관례) |
| `eef_rot6d` | `[6]` | 6D rotation representation |
| `gripper_opening` | `[1]` | 정규화된 gripper 개방도 |
| `gripper_qpos` | `[2]` | 2-finger gripper joint position |
| `gripper_action` | `[1]` | 직전 gripper 명령 (`{-1, +1}` 등) |
| `joint_pos` | `[7]` | Arm joint position |
| `joint_vel` | `[7]` | Arm joint velocity |
| `base_pos` | `[3]` | 모바일 로봇 베이스 위치 |
| `base_quat` | `[4]` | 모바일 로봇 베이스 회전 |

### 3.3 지시문 — `task`
- 타입: `str`. Natural language instruction.

---

## 4. Action 스키마 (Sub-key)

### 4.1 표준 Sub-key

| 키 | shape | 의미 |
|---|---|---|
| `action.eef_pos` | `[N, 3]` | EEF translation (relative delta 또는 absolute pos) |
| `action.eef_euler` | `[N, 3]` | EEF rotation Euler |
| `action.eef_rot6d` | `[N, 6]` | EEF rotation 6D |
| `action.eef_quat` | `[N, 4]` | EEF rotation Quaternion (xyzw) |
| `action.gripper` | `[N, 1]` | Gripper scalar (모델 native range) |
| `action.joint_pos` | `[N, 7]` | Joint space target |

`N = n_action_steps`. 모델이 action chunk를 반환하면 `N > 1`.

### 4.2 `action_type` 관례

- **`"relative"`** — EEF는 현재 포즈에 더하는 delta. Gripper는 부호 기반 이산화 (threshold=0.0).
- **`"absolute"`** — EEF는 월드/로봇 좌표계의 절대 target. Gripper threshold 관례는 모델별로 다를 수 있음 (X-VLA는 0.8).

**중요**: `action_type`은 서버 → 벤치 간 해석 계약일 뿐, 한 벤치마크가 둘 다 지원하는 경우가 있다 (Calvin은 relative/absolute 모두 지원, RoboCasa는 relative 한정).

### 4.3 Rotation 우선순위 (Calvin ActionProcessor 기준)
```
eef_euler > eef_rot6d > eef_quat
```
- `eef_rot6d` → Gram-Schmidt 직교화로 회전행렬 복원 → Z-Y-X Euler 추출.
- `eef_quat`(xyzw) → arctan2/arcsin 기반 Euler 변환.

---

## 5. 클라이언트 — `VLAClient`

`scripts/utils/vla_client.py` — 모든 벤치마크가 공유하는 단일 클라이언트.

### 5.1 시그니처
```python
class VLAClient:
    def __init__(self, url: str, timeout: float = 60.0): ...
    def health_check(self) -> dict | None: ...
    def wait_until_ready(self, max_wait: float = 180.0, poll_interval: float = 3.0) -> dict: ...
    def reset(self) -> None: ...
    def predict(
        self,
        images: dict[str, np.ndarray],        # {camera_name: HWC uint8}
        states: dict[str, np.ndarray] | None, # {state_field: ndarray}
        instruction: str,
    ) -> tuple[np.ndarray | dict, float]:
        """returns (action_or_subkey_dict, latency_ms)"""
```

### 5.2 동작 요약
1. `images` dict를 순회하여 각 ndarray를 base64 PNG로 encode하고 `observation.images.{k}` 키로 packing.
2. `states` dict의 각 ndarray를 list로 변환하여 `observation.state.{k}` 키로 packing.
3. `task = instruction` 추가 후 `POST /act`.
4. 응답에 `action.*` 키가 하나라도 있으면 **sub-key dict** 그대로 반환, 없으면 flat `action`(2D array)을 반환.
5. 1D 응답은 자동으로 2D로 승격.

---

## 6. 현재 등록된 모델 서버

| 모델 | 포트 | action_type | 반환 sub-key | n_steps | 주요 특징 |
|---|---|---|---|---|---|
| **X-VLA** (`serve_xvla.py`) | 8100 | absolute | `eef_pos`, `eef_rot6d`, `gripper` | 30 | 20D dual-arm; euler→rot6d 자체 변환 |
| **DreamVLA** (`serve_dreamvla.py`) | 8200 | relative | `eef_pos`, `eef_euler`, `gripper` | 1 | History-based; quat→euler fallback |
| **UP-VLA** (`serve_upvla.py`) | 8300 | relative | `eef_pos`, `eef_euler`, `gripper` | `act_step` | 이미지만 필수, state 불필요 |
| **LeRobot (pi0/pi05)** (`serve_lerobot.py`) | 8400 | relative | `eef_pos`, `eef_euler`, `gripper` 또는 flat `action` | policy config | `STATE_KEY_ORDER`로 state 순서 고정 |
| **GR00T N1.6** (`serve_groot.py`) | 8500 | relative | `base_motion`, `control_mode`, `end_effector_position`, `end_effector_rotation`, `gripper_close` | `delta_indices` | Embodiment별 동적 action dim |

### 6.1 서버의 obs key fallback 패턴
- **Quat → Euler**: `_quat_xyzw_to_euler()` (DreamVLA)
- **Euler → Rot6D**: `_euler_to_rot6d()` (X-VLA, Z-Y-X)
- **Gripper**: `gripper_action` > `gripper_opening` > `gripper_qpos` > `[0.0]` default
- 없는 키는 0 벡터로 채우거나 skip.

---

## 7. 벤치마크 측 Processor (`src/processor/`)

### 7.1 기본 구조
```
ProcessorStep (base.py)
├── ObservationProcessorStep  — process_observation(obs) -> obs
└── ActionProcessorStep       — process_action(action) -> action

DataProcessorPipeline          — 여러 Step을 순차 조립 (LeRobot 호환)
```

### 7.2 Factory (`src/processor/factory.py`)
```python
make_calvin_processors(
    use_wrist: bool,
    gripper_threshold: float,  # relative=0.0, absolute=0.8
    action_type: str,          # "relative" | "absolute"
) -> (obs_pipeline, action_pipeline)

make_robocasa_processors() -> (obs_pipeline, action_pipeline)
```

현재 등록된 벤치마크: **Calvin**, **RoboCasa**. (LibERO는 별도 경로 — LeRobot native eval 사용.)

### 7.3 Calvin Obs
- `rgb_static` (CHW, [-1,1]) → `observation.images.static` (HWC uint8)
- `rgb_gripper` → `observation.images.wrist` (옵션)
- `robot_obs` 15D 파싱:
  | range | unified key |
  |---|---|
  | `[0:3]` | `observation.state.eef_pos` |
  | `[3:6]` | `observation.state.eef_euler` |
  | `[6:7]` | `observation.state.gripper_opening` |
  | `[7:14]` | `observation.state.joint_pos` |
  | `[14:15]` | `observation.state.gripper_action` (`{-1,+1}`) |

### 7.4 Calvin Action
- 회전 우선순위: `eef_euler > eef_rot6d > eef_quat`.
- Gripper 이산화: `gripper_val < threshold ? +1(open) : -1(close)`.
- **Relative** 출력: `np.ndarray` shape `[7]` = `[pos(3), euler(3), gripper(1)]`.
- **Absolute** 출력: `tuple(pos, euler, gripper)` 3-tuple (Calvin env가 absolute 인자를 직접 받음).

### 7.5 RoboCasa Obs
- Cameras: `{robot_prefix}agentview_left_image` → `static`, `{robot_prefix}eye_in_hand_image` → `wrist`.
- State: 명시적 키 우선, 없으면 68D `proprio-state`에서 복원:
  - `eef_pos(3)`, `eef_quat(4)`, `gripper_qpos(2)`, `joint_pos/vel(7+7)`, `base_pos/quat(3+4)`.

### 7.6 RoboCasa Action
- 입력: VLA의 7D `[arm(6), gripper(1)]`.
- 출력: 12D `[arm(6), gripper(2 복제), base(3), torso(1)]` — PandaMobile 환경 맞춤.
- Gripper 값은 2-finger 양쪽에 broadcast. Base/torso는 0 패딩 (mobile 미사용).
- `action_type`은 `"relative"`만 지원.

---

## 8. Action Chunk 처리 규칙

서버가 `n_action_steps > 1`로 chunk를 반환하는 경우, 벤치마크 eval 루프가 이를 어떻게 소화할지는 벤치마크마다 다르다.

| 벤치마크 | 정책 |
|---|---|
| **RoboCasa** (`robocasa_vla_eval.py`) | 첫 스텝만 실행 (`actions[0]`). 매 env step마다 재예측. |
| **Calvin** (`calvin_eval.py`) | `action_buffer` 캐싱: chunk를 받아 소진될 때까지 한 스텝씩 꺼내 env에 먹인 후, 소진 시 재예측. `server_info["n_action_steps"]`로 버퍼 길이 결정. |
| **LibERO** (`eval_pi05_libero.py`) | LeRobot native `eval_policy_all` 사용 (통일 프로토콜을 거치지 않는 별도 경로). |

---

## 9. End-to-End 데이터 흐름 예시 (Calvin + X-VLA)

```
1. Calvin env.step → obs = {"rgb_obs": {"rgb_static": CHW[-1,1]}, "robot_obs": 15D, ...}

2. CalvinObsProcessor:
   → {"observation.images.static": HWC uint8,
      "observation.state.eef_pos": [x,y,z],
      "observation.state.eef_euler": [r,p,y],
      "observation.state.gripper_opening": [g],
      "observation.state.joint_pos": [...],
      "task": "open the drawer"}

3. VLAClient.predict → POST /act (base64 PNG로 이미지 인코딩)

4. X-VLA server:
   - needs eef_pos (있음), eef_rot6d (없음 → euler로부터 자체 변환), gripper_opening (있음)
   - returns {"action.eef_pos":   [[x,y,z]×30],
              "action.eef_rot6d": [[6D]×30],
              "action.gripper":   [[g]×30]}

5. VLAClient → sub-key dict 그대로 반환 (+ latency_ms)

6. Calvin eval loop: action_buffer에 저장, 한 스텝씩 꺼냄

7. CalvinActionProcessor (action_type="absolute", threshold=0.8):
   - rot6d → Gram-Schmidt → Z-Y-X euler
   - gripper: val < 0.8 ? +1 : -1
   → (pos, euler, gripper) 3-tuple

8. Calvin env.step(3-tuple) 실행
```

---

## 10. 새 모델/벤치 추가 레시피

### 10.1 새 모델 추가
1. `docker/<model>/Dockerfile` 작성, `docker-compose.yml`에 서비스 등록 (`network_mode: host`, GPU 할당).
2. `scripts/serve_<model>.py` 작성:
   - `GET /health` — `action_type`, `action_keys`, `n_action_steps` 노출 필수.
   - `POST /reset` — 히스토리 초기화.
   - `POST /act` — `observation.*` 중 필요한 키만 꺼내 쓰고, 없는 키는 자체 변환/기본값. 응답은 `action.*` sub-key로.
3. (선택) 학습용 `src/datasets/adapters/<model>.py` — LeRobotDataset wrapping + collator.

### 10.2 새 벤치마크 추가
1. `src/processor/obs/<bench>.py` — env obs → `observation.*` 통일 key로 변환하는 `ObservationProcessorStep`.
2. `src/processor/action/<bench>.py` — `action.*` sub-key dict → env action으로 조립하는 `ActionProcessorStep`.
3. `src/processor/factory.py`에 `make_<bench>_processors()` 등록.
4. `scripts/<bench>_eval.py` 작성 — `VLAClient` + factory 사용. Action chunk 정책 결정 (single-step 실행 vs. buffer caching).

---

## 11. Docker 구성 요약

`docker-compose.yml` 주요 서비스:

| 서비스 | Python | 역할 |
|---|---|---|
| `robocasa` | 3.11 | RoboCasa 시뮬레이션 + 평가 (+ KasmVNC GUI) |
| `calvin` | 3.8 | Calvin 벤치마크 (headless) |
| `xvla` | 3.10 | X-VLA 서버 (:8100) |
| `dreamvla` | 3.10 | DreamVLA 서버 (:8200) |
| `upvla` | 3.10 | UP-VLA 서버 (:8300) |
| `lerobot` | 3.10 | pi0/pi05 서버 (:8400) |
| `groot` | 3.10 | GR00T N1.6 서버 (:8500) |

- 모든 컨테이너 `network_mode: host` → localhost HTTP로 통신.
- `/temporal_vla` 볼륨 공유 (공통 코드, 체크포인트, 출력).
- `VLAClient(url=...)`의 URL만 바꾸면 모델 스왑.

---

## 12. 핵심 파일 레퍼런스

| 파일 | 역할 |
|---|---|
| `scripts/utils/vla_client.py` | 통일 HTTP 클라이언트 |
| `scripts/serve_xvla.py` | X-VLA 서버 (absolute, rot6d) |
| `scripts/serve_dreamvla.py` | DreamVLA 서버 (relative, euler) |
| `scripts/serve_upvla.py` | UP-VLA 서버 (relative) |
| `scripts/serve_lerobot.py` | pi0/pi05 서버 |
| `scripts/serve_groot.py` | GR00T 서버 |
| `src/processor/base.py` | ProcessorStep 추상 기본 |
| `src/processor/factory.py` | 벤치마크별 pipeline 빌더 |
| `src/processor/obs/calvin.py` | Calvin obs → 통일 키 |
| `src/processor/obs/robocasa.py` | RoboCasa obs → 통일 키 |
| `src/processor/action/calvin.py` | 통일 sub-key → Calvin action |
| `src/processor/action/robocasa.py` | 통일 sub-key → RoboCasa 12D action |
| `scripts/calvin_eval.py` | Calvin eval loop (action_buffer) |
| `scripts/robocasa_vla_eval.py` | RoboCasa eval loop (single-step) |
| `docker-compose.yml` | 컨테이너 토폴로지 |

---

## 13. 한 줄 요약

> **관측은 `observation.{images|state}.*` dot-key로, 액션은 `action.*` sub-key + `action_type` 메타로 협상한다.**
> 이 계약만 지키면 모델과 벤치마크는 서로를 전혀 모른 채 조합될 수 있다.
