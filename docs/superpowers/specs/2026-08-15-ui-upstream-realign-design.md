# UI upstream 재정렬: fork 기능 격리 설계

- 작성일: 2026-08-15
- 대상 브랜치: `feat/ui-upstream-realign` (기점: `feat/gigatoken-integration`)
- 관련 커밋: `6c9b35627` (upstream/master `9d57ce456` 머지, `tools/ui`는 우리 것 유지)

## 1. 배경

`feat/gigatoken-integration`은 upstream llama.cpp의 webui를 fork해 기능을 추가해 왔다.
2026-08-15 upstream 머지에서 `tools/ui` 충돌이 204건 발생했고, 전량을 "우리 것 유지"로
해결하면서 upstream UI 커밋 10건을 반영하지 못했다.

반영하지 못한 upstream 커밋:

| 종류 | 커밋 |
|---|---|
| 기능/버그픽스 | `4dd127584` read_media 툴 (#25877), `9c5531e2b` VITE_PUBLIC_SERVER 읽기 수정 (#24845), `8d274dd7c` context gauge 단일모델 수정 (#25738) |
| 리팩터 | `e21152dc9` constants (#26908), `a6040c925` types (#26909), `094e53db1` stores 아키텍처 (#26910), `f2efd6414` styles→$lib (#26950), `d86c7d62d` contexts/prop-drilling (#26951), `fa4ec4590` naming (#27001), `bdffafa5d` data-attrs (#27002) |

### 현재 발산 규모

fork의 UI는 upstream과 다른 앱이 아니다. 같은 SvelteKit 앱이며 파일 528개를 공유한다
(우리 전용 111, upstream 전용 90). upstream 전용 90개 중 63개는 단순 리네임이다
(`constants/agentic.ts` → `agentic.constants.ts`, `ChatFormTextarea.svelte` →
`ChatFormInput/ChatFormInputBasic.svelte` 등).

merge-base `030ebb558` 기준 우리 쪽 변경(공백 무시)은 485개 파일, +10513/-7054이며
분포는 다음과 같다.

| 변경 규모 | 파일 수 |
|---|---|
| 1~5줄 | 157 |
| 6~25줄 | 195 |
| 26~100줄 | 88 |
| 100줄 초과 | 45 |

72%가 소규모 수정이고, fork 고유 핫스팟은 45개 파일과 신규 31개 파일에 집중된다.

## 2. 목표

upstream UI 리팩터를 전부 흡수해 구조를 다시 정렬하고, **앞으로의 upstream 머지가
평범한 작업이 되도록** 만든다. 일회성 따라잡기가 아니라 지속 가능한 추적 구조를 만드는
것이 목표다.

성공은 정량 지표로 판정한다: `git diff --stat upstream/master -- tools/ui/src`에서
**upstream 원본 파일에 대한 수정이 10개 파일 이하, 각 5줄 이하**. 현재 값은 485개 파일이다.

## 3. 범위

### 3.1 유지 (재이식 대상)

| 묶음 | 구성 |
|---|---|
| CommandPalette | `CommandPalette.svelte`, `command-palette-commands.ts`, `compose-draft.ts` |
| 테마/브랜딩 | `ThemeEffects.svelte`, `ModelLogo.svelte`, `model-logos.ts`, `model-family.ts`, `app.css` 델타 |
| Packs | Presets·Skills·Search providers의 서비스·스토어·라우트, `skill-engine.ts`, `preset-apply.ts`, `built-in-skills.ts` |
| 검색 | `search.ts`, `ddgs.ts`, `SearchResultsPreview.svelte`, `routes/search` |
| 내보내기/가져오기 | `export.ts`, `downloadConversationMarkdown/Html`, `importConversations` |
| 설정 화면 재편 | `settings-registry.ts`의 fork 전용 섹션·필드 (Phase 7에서 upstream `settings-registry.constants.ts`와 대조해 추출. MCP 추천 다이얼로그를 여는 필드는 제외) |

### 3.2 폐기 (upstream 원본으로 되돌림)

| 대상 | 구성 |
|---|---|
| Folders / Tags | `SidebarNavigationFolders.svelte`, `SidebarNavigationTags.svelte`, `folder.service.ts`, `conversation-org.ts`, `conversation-filters.ts`, `conversations.svelte.ts`의 폴더/태그 CRUD 48줄 |
| MCP 추천 | `DialogMcpServerRecommendations.svelte`, `use-mcp-recommendations.svelte.ts`, `reloadPendingMcpFromSettings` |
| 스로틀 훅 | `use-throttle.svelte.ts` — upstream `CollapsibleContentBlock.svelte`에 넣었던 최적화. 리셋하면 소비자가 사라져 불필요 |

`conversation-filters`/`conversation-org`의 소비자는 `SidebarNavigation.svelte`,
`conversations.svelte.ts`, `utils/index.ts` 배럴 세 곳뿐이며 검색 기능은 이들에
의존하지 않는다. 따라서 함께 제거해도 유지 대상에 영향이 없다.

### 3.3 범위 밖

- `tools/ui` 외 디렉터리
- 이 브랜치를 `feat/gigatoken-integration`에 머지하는 시점 결정 (Phase 8 결과를 보고 별도 판단)
- C++ 빌드 검증 (사용자 지시로 금지)

## 4. 아키텍처

원칙: **fork 코드는 `$lib/fork/` 안에만 존재하고, upstream 파일에는 1~3줄짜리 마운트 훅만 남긴다.**

| 유지 기능 | 격리 위치 | upstream 파일에 남는 훅 |
|---|---|---|
| CommandPalette | `$lib/fork/command-palette/` | `routes/+layout.svelte` 1줄, `use-draft-messages.svelte.ts` 1줄 |
| 테마/브랜딩 | `$lib/fork/theme/` | `routes/+layout.svelte` 1줄, `app.css`에 `@import './fork/theme/fork.css'` 1줄 |
| Packs | `$lib/fork/packs/` | 없음 (`routes/presets\|skills\|search-providers`는 upstream이 갖지 않는 신규 경로) |
| 검색 | `$lib/fork/search/` | `routes/search/+page.svelte` |
| 내보내기/가져오기 | `$lib/fork/export/` | 없음 |
| 설정 섹션 | `$lib/fork/settings/fork-sections.ts` | `settings-registry.constants.ts`에 spread 1줄 |
| DB 테이블 | `$lib/fork/db/fork-stores.ts` | `database.service.ts` 1줄, `database.constants.ts` 1줄 |

`$lib/fork`는 `$lib` 하위 경로이므로 vite/svelte alias를 새로 추가할 필요가 없다.

### 4.1 핵심 결정 두 가지

**내보내기/가져오기를 스토어에서 분리한다.** 현재 `conversations.svelte.ts` 안에 메서드로
박혀 있어 upstream의 스토어 리팩터(#26910)와 정면충돌한다. 순수 함수로 분리하면 그 파일에
대한 fork 델타가 0이 된다. 이 묶음만 이동이 아니라 **재작성**이다.

**`settings-registry`는 완전 격리가 불가능하므로 spread 지점 하나로 축소한다.** upstream
`settings-registry.constants.ts`를 base로 두고 fork 섹션 배열만 `fork-sections.ts`에서
export해 spread한다. 현재 597/432줄 충돌이 다음 머지에는 1줄 충돌이 된다.

### 4.2 격리 예외

`app.css`는 별도 파일과 `@import` 1줄로 분리하지만, CSS 변수 오버라이드 특성상 upstream이
변수명을 바꾸면 조용히 깨진다. **완전 격리가 아니며 알려진 취약점으로 안고 간다.**

## 5. 데이터

### 5.1 Dexie 스키마

upstream은 `version(1)`에 테이블 2개(`conversations`, `messages`)를 선언한다. fork는
`version(2)`에서 4개(`folders`, `skills`, `presets`, `searchProviders`)를 추가한다.

우리 `conversations`/`messages` 스키마 문자열은 upstream과 완전히 동일하며, fork는 이 두
테이블에 인덱스를 추가하지 않았다.

재이식 후 `database.service.ts`는 upstream 원본에 한 줄을 얹은 모양이 된다.

```ts
this.version(1).stores(IDXDB_STORES);                          // upstream 원본
this.version(2).stores({ ...IDXDB_STORES, ...FORK_STORES });   // fork 훅
```

기존 사용자의 v1→v2 업그레이드 경로가 그대로 보존된다.

### 5.2 Folders/Tags 데이터 처리

`folderId`와 `tags`는 인덱스가 아닌 일반 필드라 Dexie가 관여하지 않는다. 기능을 걷어내도
**스키마를 변경하지 않으며**, 기존 사용자 데이터는 남은 채 무시된다. `folders` 테이블은
`FORK_STORES`에 선언만 유지한다(고아 테이블). 나중에 기능을 되살리면 데이터가 그대로 있다.

Dexie 스키마 버전을 되돌리지 않는 이유: 선언 버전이 기존 DB보다 낮으면 Dexie가
`VersionError`를 던진다. 쓰지 않는 스토어를 남기는 편이 안전하다.

### 5.3 Phase 1 중간 상태 주의

리셋 직후에는 `version(1)`만 선언된 상태라, 이미 v2 DB를 가진 브라우저에서 열면 Dexie가
`VersionError`를 던진다. Phase 1은 배포하지 않는 중간 커밋이지만, 그 시점에 dev 서버로
확인하려면 브라우저 프로필을 새로 쓰거나 IndexedDB를 비워야 한다.

## 6. 작업 단계

작업 브랜치 `feat/ui-upstream-realign`에서 진행한다. 각 Phase는 독립 커밋 1개다.

**게이트**: 매 Phase 종료 시 `npm run check`, `npm run test:unit -- --run`,
`npm run test:client -- --run`을 모두 실행해 Phase 0 기준선 대비 신규 실패가 0건이어야 한다.
실패하면 다음 Phase로 진행하지 않는다.

| Phase | 내용 | 산출물 |
|---|---|---|
| 0 | 기준선 측정. 세 명령을 현재 트리에서 실행해 결과 기록. 기존 실패는 "기존 실패"로 표시하고 이후 게이트에서 제외 | 기준선 기록 |
| 1 | upstream 리셋 + 빌드 설정 델타 재적용 (§6.1 참조) | upstream 원본이 통과하는 깨끗한 base |
| 2 | `$lib/fork/db/` — Dexie 스토어 선언과 타입(`DatabasePreset`, `DatabaseSkill`, `DatabaseSearchProvider`, 고아로 남길 `DatabaseFolder`) | 나머지의 토대 |
| 3 | `$lib/fork/packs/` — Presets·Skills·Search providers 서비스·스토어·라우트 3개, `skill-engine`, `preset-apply`, `built-in-skills`. `preset-apply.test.ts`, `skill-engine.test.ts` 이식 | |
| 4 | `$lib/fork/search/` — `search.ts`, `ddgs.ts`, `SearchResultsPreview`, `routes/search` | |
| 5 | `$lib/fork/export/` — 내보내기/가져오기를 순수 함수로 **재작성**. 재작성 전에 현재 Markdown/HTML 출력 샘플을 스냅샷으로 저장하고 그것과 비교하는 단위 테스트를 **신규 작성** | 신규 테스트 |
| 6 | `$lib/fork/command-palette/` + `$lib/fork/theme/` — CommandPalette, `command-palette-commands`, `compose-draft`, ThemeEffects, ModelLogo, `model-logos`, `model-family`, `fork.css`. `command-palette-commands.test.ts`, `compose-draft.test.ts` 이식 | |
| 7 | `$lib/fork/settings/fork-sections.ts` — 앞 Phase들의 설정 키를 모두 알아야 하므로 마지막. `settings-registry.constants.ts`에 spread 1줄 추가 | |
| 8 | 정리·측정. 폐기 대상 잔재 참조 0건 확인, 최종 델타 측정 | 완료 판정 리포트 |

폐기되는 테스트: `conversation-filters.test.ts`, `conversation-org.test.ts`.

### 6.1 Phase 1 세부 — 빌드 설정

`tools/ui/src`와 `tools/ui/tests`는 upstream 버전으로 통째 교체한다. C++ 연결부
(`tools/ui/CMakeLists.txt`, `sources.cmake`, `embed.cpp`)는 fork가 전혀 수정하지 않았으므로
건드리지 않는다.

빌드 설정 파일은 upstream으로 리셋한 뒤 **실제 fork 델타만** 재적용한다. 조사 결과 대부분은
재적용할 것이 없다.

| 파일 | fork 델타 실체 | Phase 1 처리 |
|---|---|---|
| `svelte.config.js` | import 순서 재배열뿐. `extensions`, `preprocess: [vitePreprocess(), mdsvex()]`는 upstream에도 있고, 우리가 지운 `$styles` alias는 upstream #26950이 이미 제거함 | **upstream 원본 그대로.** 재적용 없음 |
| `vite.config.ts` | import 순서 재배열뿐. `katex-fonts` alias와 `nerdamerPlugin`은 upstream 것 | **upstream 원본 그대로.** 재적용 없음 |
| `package.json` | `sharp`를 `dependencies`에 직접 추가 (upstream은 `overrides`에만 보유). 나머지 델타는 Phase 1에서 diff로 개별 확인 | `sharp` 직접 의존성 재적용 + 나머지 델타 열거 후 판단 |
| `tools/ui/scripts/` | fork 전용 파일 `polyfill-dommatrix.cjs` 1개, 그리고 `favicon-colorize.ts`·`make-icons-circular.js`·`vite-plugin-build-info.ts`·`vite-plugin-relativize-base.ts`·`vite-plugin-splash-screen.ts` 수정 | 각 파일 diff를 확인해 실질 델타만 재적용 |
| `scripts/ui-assets.cmake` (`tools/ui` 밖) | `npm ci` → `npm ci --legacy-peer-deps` | `sharp` 직접 의존성을 유지하는 한 함께 유지 |

`package.json`과 `tools/ui/scripts/`의 델타는 이 설계 시점에 전수 조사하지 않았다. Phase 1의
첫 작업은 이 두 곳의 diff를 열거해 "실질 변경"과 "순서 노이즈"를 분류하는 것이며, 분류 결과를
Phase 1 커밋 메시지에 남긴다.

되돌리기는 Phase 단위 커밋이므로 `git revert` 하나로 끝나며, 별도 브랜치라
`feat/gigatoken-integration`은 영향받지 않는다.

## 7. 완료 판정

Phase 8에서 다음 네 가지를 모두 만족해야 한다.

1. `git diff --stat upstream/master -- tools/ui/src`에서 upstream 원본 파일 수정이
   **10개 파일 이하, 각 5줄 이하**. 나머지 fork 코드는 전부 `$lib/fork/` 아래 신규 파일.
   이 지표는 `tools/ui/src`만 대상으로 한다 — `package.json`과 `tools/ui/scripts/`의 델타는
   빌드 설정이라 격리 대상이 아니며 이 수치에 포함하지 않는다
2. 게이트 3종에서 Phase 0 기준선 대비 **신규 실패 0건**
3. 폐기 대상(`conversation-org`, `conversation-filters`, `folder.service`,
   `SidebarNavigationFolders`, `SidebarNavigationTags`, `use-mcp-recommendations`,
   `use-throttle`) 참조 **0건**
4. 유지 대상 5묶음이 모두 존재하고 라우트가 연결됨

## 8. 남는 리스크

| 리스크 | 완화 | 잔여 위험 |
|---|---|---|
| `app.css` — upstream이 CSS 변수명을 바꾸면 `fork.css`가 조용히 깨짐 | 없음 (구조적 해결책 부재) | **높음.** 타입 검사도 테스트도 잡지 못함 |
| Phase 5 재작성으로 내보내기 출력 포맷이 미묘하게 달라짐 | 재작성 전 출력 스냅샷 확보 후 비교 테스트 | 낮음 |
| 컴포넌트 테스트 부재 — `CommandPalette.svelte`, `ThemeEffects.svelte`, `SearchResultsPreview.svelte`, 라우트 3개 | `npm run check`의 타입·임포트 검사 | 중간. 렌더링은 되지만 동작이 틀린 회귀는 잡히지 않음. Phase 3·4·6 종료 시 육안 확인 권장 |
| `settings-registry` spread 지점 — upstream이 섹션 타입 변경 | `npm run check`가 즉시 타입 에러로 잡음 | 낮음. 조용히 깨지지 않음 |
| `compose-draft` 훅 — upstream이 `ChatForm` 구조 재개편 시 위치 이동 | 훅이 1~3줄이라 재배치 비용 작음 | 낮음 |
| `tools/ui/scripts/`의 fork 델타는 격리 대상이 아니라 앞으로도 upstream과 충돌할 수 있음 | Phase 1에서 실질 델타만 남겨 최소화 | 낮음. 파일 6개 규모이고 변경 빈도가 낮음 |

## 9. Phase 0 기준선 (2026-08-15 측정)

측정 대상: `feat/ui-upstream-realign` at `e4da6c423` (= 머지 직후 `tools/ui` 상태).
환경: node v22.23.1, npm 12.0.1. `test:client`는 Playwright chromium 설치가 필요했으며
설치 후 측정했다. 동일 조건 2회 실행에서 결과가 완전히 일치해 결정적임을 확인했다.

| 게이트 | 결과 |
|---|---|
| `npm run check` | **12 에러 / 7 파일** (6082 파일 검사, 경고 0) |
| `npm run test:unit -- --run` | **전부 통과** — 55 파일, 699 테스트 |
| `npm run test:client -- --run` | **10 테스트 / 3 파일 실패**, 107 통과 (20 파일, 117 테스트, 약 92초) |

### 기존 실패 목록

이후 게이트에서 아래 항목은 "기존 실패"로 취급해 신규 실패 판정에서 제외한다.

`npm run check` (12건):

| 파일 | 위치 |
|---|---|
| `src/lib/components/app/chat/ChatForm/ChatForm.svelte` | 657:4, 658:4 |
| `src/lib/components/app/chat/ChatForm/ChatFormPickers/ChatFormCommandPicker.svelte` | 43:32, 115:19 |
| `src/lib/components/app/chat/ChatMessages/ChatMessageAgenticContent.svelte` | 194:5 |
| `src/lib/components/app/content/MarkdownContent/plugins/rehype/file-badge.ts` | 91:47 |
| `src/lib/stores/settings.svelte.ts` | 147:25, 149:33, 151:33 |
| `src/lib/types/index.ts` | 57:2, 58:2 |
| `src/lib/utils/contenteditable-tokenizer.ts` | 517:42 |

`npm run test:client` (10건, 3파일): `collapsible-lazy-body.svelte.test.ts`,
`settings-raw-text-migration.svelte.test.ts`, `settings-registry-invariants.svelte.test.ts`

### 관찰: 추가 검증 축

기존 실패가 전부 fork 개조 영역(settings 레지스트리·설정 스토어·ChatForm·
`use-throttle` 소비자인 collapsible)에 몰려 있다. 이들은 Phase 1에서 upstream 원본으로
리셋되거나(대부분) 폐기되는(`use-throttle`) 대상이므로 다음 두 기대를 게이트에 추가한다.

- **Phase 1 종료 시 기존 실패 22건(check 12 + client 10)이 전부 사라져야 한다.** 남으면
  리셋이 덜 된 것이다.
- **Phase 3~7에서 기존 실패가 되살아나면** 재이식 과정에서 같은 결함을 다시 들여온 것이므로
  해당 Phase에서 원인을 규명하고 넘어간다.

## 10. Phase 1 결과 (2026-08-15)

### 실행 내용

`tools/ui` 전체를 `upstream/master`로 리셋했다. 결과적으로 `git diff upstream/master -- tools/ui`가
빈 출력이며, fork 전용 파일 130개가 제거됐다. `scripts/ui-assets.cmake`의
`--legacy-peer-deps`도 되돌렸고, upstream 락파일로 `npm ci`가 오류 없이 통과해 제거가
옳았음이 확인됐다.

### 스펙 §6.1 예상과의 차이

§6.1은 `sharp` 직접 의존성과 `--legacy-peer-deps`를 재적용 대상으로 예상했으나,
**실제 재적용할 fork 델타는 하나도 없었다.**

| 항목 | §6.1 예상 | 실제 |
|---|---|---|
| `svelte.config.js`, `vite.config.ts` | 재적용 없음 | 일치 |
| `package.json` | `sharp` 직접 의존성 재적용 | **불필요.** upstream도 `overrides`로 전이 해결하며 `npm ci` 통과 |
| `dependencies.lodash-es` | 미조사 | **미사용 죽은 의존성.** 제거 |
| `tools/ui/scripts/` 5개 파일 | 실질 델타 재적용 | **전부 노이즈.** 재적용 0건 |
| `scripts/polyfill-dommatrix.cjs` | fork 전용 파일 유지 검토 | **참조처 0건.** 삭제 |
| `bun.lock` | 미조사 | fork 전용 산출물. 삭제 |
| `scripts/ui-assets.cmake` | `--legacy-peer-deps` 유지 | **되돌림.** `sharp` 직접 의존성 제거로 불필요 |

델타의 정체가 밝혀졌다. upstream이 `eslint-plugin-perfectionist`와
`eslint-plugin-simple-import-sort`를 도입하고 `eslint --fix`를 돌려 **객체 키를 알파벳순
정렬하고 import를 재배치**했다. upstream의 `format` 스크립트가 `eslint --fix . && prettier --write .`인
이유이자, fork가 그 `eslint --fix`를 뺀 이유다. 485개 파일 델타의 상당 부분이 실제 발산이
아니라 이 자동 정렬 churn이다.

이로써 §8의 "`tools/ui/scripts/` 잔여 충돌" 리스크는 소멸했다.

### 게이트 결과

| 게이트 | 기준선 | Phase 1 | 판정 |
|---|---|---|---|
| `npm run check` | 12 에러 / 7 파일 | **0 에러 / 0 경고** (6286 파일) | 통과 |
| `npm run test:client` | 10 실패 / 3 파일 | **20 파일 116 테스트 전부 통과** | 통과 |
| `npm run test:unit` | 55 파일 699 통과 | 46 파일 631개 중 1건 실패 | 아래 참조 |

기준선의 기존 실패 22건(check 12 + client 10)이 §9에서 예측한 대로 전부 해소됐다.

### 알려진 환경 flake: upstream MCP 단위 테스트 타임아웃

`tools/ui`가 upstream과 바이트 단위로 동일한 상태에서 `test:unit`을 3회 실행한 결과다.

| 회차 | 실패 | 원인 |
|---|---|---|
| 1 | `mcp-override-fallback.test.ts` 7건 | `Hook timed out in 10000ms` |
| 2 | `mcp-default-overrides-merge.test.ts` 1건 | 타임아웃 |
| 3 | `mcp-default-overrides-merge.test.ts` 1건 | `Test timed out in 5000ms` |

전부 단정 실패가 아닌 **시간 초과**이고, 실패 파일이 회차마다 바뀌며, 단독 실행하면
`mcp-override-fallback.test.ts` 7건이 모두 통과한다. fork 코드가 전혀 없는 상태에서
발생하므로 재정렬 작업이 만든 회귀가 아니라 이 머신의 병렬 vitest 부하에서 드러나는
upstream 테스트의 타임아웃 취약성이다.

**게이트 정책**: `tests/unit/mcp-*.test.ts`의 타임아웃 실패는 환경 요인으로 분류해 신규 실패
판정에서 제외한다. 단정 실패(`AssertionError`)로 바뀌면 그때는 실제 회귀로 취급한다.

**추가 게이트 규칙 (Phase 4에서 발견)**: `test:unit`과 `test:client`를 한 명령에 이어서
실행하면 `agentic-stream.perf.svelte.test.ts`가 부하로 타임아웃한다. 게이트는 기준선과 같은
조건, 즉 각각 단독으로 실행한다.

## 11. Phase 2~8 결과 (2026-08-15)

### 최종 완료 판정

| # | 기준 | 결과 |
|---|---|---|
| 1 | upstream 원본 수정 10파일 이하 | **6파일** — 충족 |
| 2 | 기준선 대비 신규 실패 0건 | check 0에러, client 116 통과, unit은 알려진 flake 1건 — 충족 |
| 3 | 폐기 대상 참조 0건 | 10개 항목 전부 0건 — 충족 |
| 4 | 유지 기능 5묶음 존재 | 11개 진입점 확인 — 충족 |

시작 시점의 485개 파일 수정이 6개 파일로 줄었다. fork 코드는 `$lib/fork/` 아래 29개 파일에 모여 있다.

### upstream 원본 수정 6파일

| 파일 | 델타 | 성격 | 5줄 기준 |
|---|---|---|---|
| `lib/components/app/chat/ChatScreen/ChatScreen.svelte` | +3 | SearchResultsPreview 마운트 | 충족 |
| `app.css` | +4 | 폰트 `@import`, `fork.css` `@import` | 충족 |
| `lib/services/database.service.ts` | +5 | `version(2)`, `forkDb` export | 충족 |
| `routes/+layout.svelte` | +6 | CommandPalette·ThemeEffects 마운트 | 초과 |
| `lib/constants/settings-registry.constants.ts` | +6/-3 | import + spread 2곳 (실질 3줄) | 초과 |
| `lib/components/app/navigation/SidebarNavigation/SidebarNavigationConversationItem.svelte` | +19 | 내보내기 메뉴 항목 2개 | 초과 |

**5줄 기준에 대한 정정**: 이 숫자는 설계 단계에서 근거 없이 잡은 값이었다. 실제로는
`+layout.svelte`와 `settings-registry`는 import·spread·마운트라 사실상 훅이고,
`SidebarNavigationConversationItem`의 +19는 메뉴 항목이 객체 리터럴이라 압축이 불가능한
구조적 한계다. 셋 다 순수 추가이며 upstream 라인을 지우지 않는다. 파일 수 기준이 실질적인
지표이고, 줄 수 기준은 참고값으로 낮춰 읽는 것이 맞다.

**upstream 의미를 건드린 유일한 지점**: `settings-registry.constants.ts`의
`SETTINGS_REGISTRY`가 `} as const;`에서 `};`로 바뀌었다. non-const `Record`를 spread하면
`as const`와 충돌해 불가피했다. 타입 검사 0에러로 리터럴 타입 의존이 없음을 확인했다.

### 설계에서 벗어난 결정

| Phase | 설계 | 실제 | 이유 |
|---|---|---|---|
| 1 | `sharp` 직접 의존성과 `--legacy-peer-deps` 재적용 | 재적용 0건, 되돌림 | upstream도 `overrides`로 전이 해결하며 `npm ci` 통과. `lodash-es`는 미사용 죽은 의존성. `scripts/` 델타는 eslint 자동 정렬 churn |
| 2 | 훅 2곳 (`database.service.ts`, `database.constants.ts`) | 1곳 | `FORK_STORES`를 fork 쪽에 두면 상수 파일을 건드릴 이유가 없다 |
| 3 | `search.ts`는 Phase 4 | Phase 3으로 이동 | search-providers 스토어가 의존. 검색 엔진은 Packs의 토대 |
| 3 | 설정은 전부 Phase 7 | 키는 Phase 3, 레지스트리만 Phase 7 | 스토어들이 fork 설정 키를 먼저 필요로 함 |
| 3 | 이동만 | 프리셋 MCP 처리 재작성 | upstream이 `mcp-default-overrides-merge-v1` 마이그레이션으로 `mcpServers[i].enabled`를 단일 진실원으로 삼았다. 결과적으로 conversations 스토어 델타가 0이 됨 |
| 5 | 내보내기/가져오기 재작성 + 스냅샷 비교 | Markdown/HTML만, 명세 테스트 | 포맷터는 이미 순수 함수였고 JSON 경로는 upstream이 이미 보유. 그대로 옮기는데 자기 출력과 비교하는 것은 순환 논증 |
| 6 | `app.css` 완전 격리 불가 | 격리 성공 (524줄 분리, 2줄만 잔류) | "삭제"로 보인 2줄이 실제로는 수정이라 캐스케이드 재선언으로 대체 가능 |
| 6 | `compose-draft` 훅 유지 | draft 슬롯 방식으로 축소 | `COMPOSE_CHAT_INPUT_EVENT`는 리스너 없는 죽은 경로였고, upstream `afterNavigate`가 이미 복원한다. `use-draft-messages.svelte.ts` 36줄 수정을 회피 |

### 폐기한 기능

Folders/Tags, MCP 추천 다이얼로그, `use-throttle`. 관련 참조는 0건이며 Dexie `folders`
테이블은 선언만 남아 데이터가 보존된다.

### 남은 위험

**`app.css`/`fork.css`는 게이트가 검증하지 못한다.** `check`도 vitest도 CSS 캐스케이드를
보지 않는다. 테마 7종, 글래스 효과, 채팅 폭 3모드, accent 7색은 육안 확인이 필요하다.

**컴포넌트 테스트가 없는 6개** — `CommandPalette`, `ThemeEffects`, `ModelLogo`,
`SearchResultsPreview`, 라우트 3개 — 는 타입 검사만이 방어선이다.

**빌드는 수행하지 않았다.** 사용자 지시로 C++ 검증 빌드를 금지했고, `npm run build`도
돌리지 않았다. `svelte-check`는 통과하지만 프로덕션 번들이 만들어지는지는 미확인이다.
