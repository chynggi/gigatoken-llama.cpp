# llama-cli 모델 로딩 진행률 바

- 날짜: 2026-08-25
- 대상: `llama-cli`
- 상태: 설계 승인됨

## 1. 문제

`llama-cli`는 모델을 로딩하는 동안 `"Loading model..."` 문구와 회전 스피너(`|/-\`)만 보여준다
(`tools/cli/cli-context.cpp:119`). 대형 모델은 로딩에 수십 초에서 수 분이 걸리는데, 사용자는
얼마나 남았는지 알 수 없고 프로세스가 멈춘 것인지 구분할 수도 없다.

정작 진행률 데이터는 이미 존재한다.

- `src/llama-model-loader.cpp:1534` 가 텐서를 올리며 `llama_progress_callback` 을 호출한다.
- `tools/server/server-context.cpp:1049` 는 `params_base.load_progress_callback` 을 **무조건**
  자신의 `load_progress_callback` 으로 덮어쓴다.
- 그 콜백은 200ms 스로틀링을 거쳐 `callback_state(SERVER_STATE_LOADING, {stages, current, value})`
  를 발행한다 (`tools/server/server-context.cpp:933-954`).
- 그런데 `callback_state` 는 **라우터의 자식 프로세스일 때만** 연결된다
  (`tools/server/server.cpp:459-463`). CLI가 띄운 단일 모델 서버에서는 구독자가 없어 값이 버려진다.

즉 값은 계산되고 있고, CLI까지 전달하는 배선만 없다.

## 2. 범위

**포함**

- `llama-cli` 가 인-프로세스 스레드로 직접 띄운 `llama-server` 의 모델 로딩 진행률
  (`tools/cli/cli-server.h` 의 `cli_server::start()` 경로)
- 텍스트 / mmproj / speculative 모델의 스테이지 구분 표시

**제외**

- `--server-base` 로 외부 서버에 접속하는 경우 — 기존 스피너를 그대로 유지한다 (동작 변화 없음)
- `-hf` / `-docker` 모델 **다운로드** 진행률 (`common_download_progress`) — 별개 관심사
- HTTP 라우트, 응답 스키마, 공개 REST API 변경

## 3. 접근법 선택

### 채택: 서버 상태 콜백을 CLI에 노출

`tools/server/server.cpp` 에 전역 setter 를 추가하고, CLI 모드일 때
`ctx_server.set_state_callback()` 에 연결한다. 서버가 이미 계산해 둔 `{stages, current, value}` 를
그대로 받으므로 스테이지 표시가 추가 비용 없이 따라오고, 200ms 스로틀링도 이미 구현되어 있다.

### 기각: `common_params.load_progress_callback` 체이닝

`server-context.cpp:1049` 의 덮어쓰기를 "기존 콜백이 있으면 함께 호출"로 바꾸는 방식.
서버 변경량은 더 작지만, text / mmproj / spec 모델이 **모두 같은 콜백으로 들어와 구분되지 않는다.**
스테이지 표시 요구사항을 만족할 수 없어 기각.

### 기각: HTTP 폴링

`/health` 의 503 응답 바디에 진행률을 싣거나 새 엔드포인트를 추가하는 방식.
`--server-base` 원격 접속까지 커버할 수 있지만 서버 공개 API 를 건드려야 하고,
이번 범위(로컬 전용)에는 과하다.

## 4. 데이터 흐름

```
[server 스레드]
llama_model_loader                                   src/llama-model-loader.cpp:1534
  → server_context_impl::load_progress_callback      기존, 200ms 스로틀
    → callback_state(SERVER_STATE_LOADING,
                     {stages, current, value})       기존
      → [신규] g_state_callback                      tools/server/server.cpp
        → cli_server::on_state()                     스냅샷 저장만, 출력 없음
              │
              │  std::mutex 로 보호되는 load_state 구조체
              ▼
[CLI 메인 스레드]
cli_server::wait_ready()  기존 200ms 폴링 루프
  → 스냅샷 읽기
  → ui::progress_bar::update()                       tools/cli/cli-ui.h
    → console::progress::update()                    common/console.cpp
```

### 스레드 안전

**서버 스레드는 콘솔에 절대 쓰지 않는다.** 이것이 이 설계의 핵심 불변식이다.

`callback_state` 는 서버 스레드에서 호출되는데, 콘솔은 CLI 메인 스레드와 스피너 스레드가
이미 공유하고 있다. 따라서 콜백은 mutex 로 보호되는 스냅샷에 저장만 하고,
그리는 일은 이미 200ms 마다 도는 `wait_ready()` 루프가 전담한다.

스피너와 진행바가 같은 줄을 다투지 않도록, 첫 진행률이 도착하면 **스피너를 먼저 정지시킨 뒤**
진행바를 시작한다.

## 5. 컴포넌트

신규 소스 파일은 테스트 하나뿐이고, 나머지는 기존 파일 수정이다.

### 5.1 `tools/server/server.cpp`

```cpp
// 전역 setter — llama_server() 호출 전에 등록해야 한다
void llama_server_set_state_callback(server_state_callback_t callback);
```

`is_run_by_cli` (= `argv == nullptr`, `server.cpp:116`) 이고 자식 프로세스가 아닐 때,
기존 child 분기(`server.cpp:459-463`)와 나란히 `ctx_server.set_state_callback()` 에 연결한다.
두 분기는 상호 배타적이므로 하나뿐인 콜백 슬롯을 두고 충돌하지 않는다.

### 5.2 `common/console.h` / `common/console.cpp`

기존 `console::spinner` 와 짝을 이루는 네임스페이스를 추가한다. `advanced_display`, `simple_io`,
`out` 이 모두 `console.cpp` 내부 static 이므로 렌더링을 여기에 두는 것이 자연스럽다.

```cpp
namespace console {
    namespace progress {
        // label 예: "Loading model (1/2 text_model)", value 는 [0,1]
        // label 이 바뀌면 새 줄에 한 번 출력하고, 이후 바 줄만 제자리 갱신한다
        void update(const std::string & label, float value);

        // 진행 중인 바 줄을 개행으로 마무리한다 (그린 것이 없으면 no-op)
        void stop();
    }

    // 순수 함수 — 테스트 대상
    std::string progress_bar_str(float value, int width, bool unicode);
}
```

내부 상태: `prog_label`(현재 라벨), `prog_active`(바 줄이 열려 있는지),
`prog_last_step`(라인 모드에서 마지막으로 출력한 10% 단계).

### 5.3 `tools/cli/cli-server.h`

```cpp
struct load_state {
    std::mutex mtx;
    bool  has_progress = false;
    std::vector<std::string> stages;
    std::string current;
    float value = 0.0f;
};
```

`cli_server` 에 `load_state` 멤버와 `on_state(server_state, const json &)` 를 추가하고,
`start()` 안에서 `llama_server_set_state_callback()` 에 등록한다.
`wait_ready()` 의 기존 200ms 루프에서 스냅샷을 복사해 렌더 콜백에 넘긴다.

`wait_ready()` 는 렌더링을 직접 하지 않고 호출자가 넘긴 콜백에 위임한다.
서버 제어 로직과 UI 를 분리해 `cli-server.h` 가 `console` / `ui` 에 의존하지 않게 한다.

### 5.4 `tools/cli/cli-ui.h`

`ui::spinner` 와 같은 RAII 모양의 래퍼를 둔다.

```cpp
struct progress_bar {
    ~progress_bar() { console::progress::stop(); }
    void update(const std::string & label, float value);
};
```

`ui::show_error()` 에 `console::progress::stop()` 호출을 추가한다
(`console::spinner::stop()` 을 부르는 것과 같은 이유).

### 5.5 `tools/cli/cli-context.cpp`

`init()` 의 로컬 모델 분기(현재 `cli-context.cpp:119`)를 다음과 같이 바꾼다.

1. 기존대로 `ui::spinner` 로 시작한다 (`"\n\nLoading model..."`)
2. `wait_ready()` 에 렌더 콜백을 넘긴다
3. 첫 진행률이 도착하면 `spinner.reset()` 후 `ui::progress_bar` 를 생성해 갱신한다
4. `wait_ready()` 가 반환되면 진행바를 정리한다

`--server-base` 분기는 손대지 않는다.

## 6. 출력 형식

스테이지 헤더는 스테이지가 바뀔 때 **한 번만** 개행과 함께 출력하고, 바 줄만 `\r` 로 제자리
갱신한다. 커서 위로 이동(`\033[A`)이 필요 없으므로 VT 처리 지원 여부와 무관하게 동작한다.

```
Loading model (1/2 text_model)
[██████████████░░░░░░░░░░]  58%      ← \r 로 갱신
```

- 바 폭 24칸 고정 (터미널 폭 감지는 하지 않는다)
- 퍼센트는 `%3d` 로 우측 정렬해 줄 길이를 일정하게 유지한다. 따라서 `\r` 갱신 시 공백 패딩이
  필요 없다
- 스테이지가 하나뿐이면 `(1/1 ...)` 를 생략하고 `Loading model` 만 표시한다
- 스테이지가 바뀌면 현재 바 줄을 개행으로 닫고 새 헤더를 출력한다

### 표시 등급

두 가지 독립적인 조건으로 결정한다.

| 조건 | 결과 |
|---|---|
| `!simple_io && is_tty` | `\r` 제자리 갱신 |
| `simple_io \|\| !is_tty` | 줄 단위 추가 출력 |
| `advanced_display` | 유니코드 `█` `░` |
| `!advanced_display` | ASCII `#` `-` |

`is_tty` 는 `out` 에 대한 `isatty(fileno(out))` 으로 판정한다
(Windows 에서는 `_isatty(_fileno(out))`; `console.cpp` 는 이미 `io.h` 와 `unistd.h` 를 포함한다).

`advanced_display` 는 `console::init()` 에 전달되는 `params.use_color` 에서 오고, 이 값은
`common/arg.cpp:1426` 에서 `tty_can_use_colors()` 로 초기화된다. **`use_color` 만으로는
`--no-color` TTY 와 파이프를 구분할 수 없으므로**, 제자리 갱신 여부는 반드시 별도의
`isatty` 검사로 결정해야 한다.

`is_tty` 는 `console::init()` 에서 한 번만 계산해 static 에 보관한다. 매 갱신마다
시스템 호출을 하지 않는다.

```
기본:       [██████░░░░]  58%
ASCII:      [######----]  58%
```

줄 단위 모드에서는 바를 그리지 않고 `update()` 에 전달된 `label` 을 그대로 써서
`"<label>: NN%"` 형태로 한 줄씩 출력한다.

```
Loading model (1/2 text_model): 20%
Loading model (1/2 text_model): 30%
Loading model (2/2 mmproj_model): 10%
```

10% 단계가 바뀔 때만 출력하고, 100% 는 항상 출력한다. 라벨이 바뀌면 단계 카운터를
초기화한다.

## 7. 에러 및 중단 처리

| 상황 | 동작 |
|---|---|
| 모델 로딩 실패 | 서버 스레드 종료 → `alive()` false → `wait_ready()` 가 false 반환. `ui::progress_bar` 소멸자가 줄을 정리한 뒤 기존 `ui::show_error()` 경로가 그대로 동작 |
| Ctrl+C | `should_stop()` true → 루프 탈출 → 소멸자 정리. 기존 스피너와 동일한 수명 규칙 |
| 스테이지 전환 시 값이 1.0 → 0.0 으로 되돌아감 | 정상. 스테이지 인덱스가 함께 바뀌므로 사용자에게 혼란을 주지 않는다. 단조 증가 클램프는 넣지 않는다 |
| `value` 가 `[0,1]` 을 벗어남 | `progress_bar_str()` 가 클램프한다 |
| `--server-base` 원격 접속 | 콜백이 등록되지 않아 스냅샷이 비어 있고 기존 스피너가 유지된다 |
| 백엔드 초기화·GGUF 메타데이터 파싱 구간 | 아직 진행률이 없으므로 스피너를 유지한다. 첫 샘플 도착 시 전환 |

## 8. 검증

### 자동

`tests/test-console-progress.cpp` 를 추가하고 `llama_build_and_test()` 로 등록한다
(`tests/CMakeLists.txt:83` 의 기존 헬퍼). `console::progress_bar_str()` 의 경계값을 검증한다.

- `0.0` → 채움 0칸
- `1.0` → 채움 `width` 칸
- `0.5`, `width=24` → 채움 12칸
- 음수 → `0.0` 과 동일
- `> 1.0` → `1.0` 과 동일
- `unicode` 플래그에 따른 문자 선택
- 어떤 입력에서도 표시 폭이 `width` 로 일정

### 수동

빌드는 MSVC, `-j 6`, 경고 없이 통과해야 한다.

1. `llama-cli -m <model>` → 바가 0→100% 로 진행하고 완료 후 줄이 깨끗이 정리됨
2. `llama-cli -m <model> --no-color` → ASCII 바가 제자리 갱신됨
3. `llama-cli -m <model> | cat` → 커서 제어 문자 없이 줄 단위 로그
4. `llama-cli -m <model> --simple-io` → 줄 단위 로그
5. mmproj 모델(`--mmproj`) → 스테이지가 `(1/2 text_model)` → `(2/2 mmproj_model)` 로 전환
6. `llama-cli --server-base <url>` → 기존 스피너 동작 그대로 (회귀 없음)
7. 로딩 중 Ctrl+C → 터미널 상태 정상 복구, 프롬프트 깨짐 없음
8. `llama-server` 단독 실행 → 진행바가 나타나지 않고 기존 로그 그대로 (회귀 없음)
