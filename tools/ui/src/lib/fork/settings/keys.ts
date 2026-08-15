/**
 * Fork-owned settings keys.
 *
 * Kept separate from upstream's `SETTINGS_KEYS` so that
 * `settings-keys.constants.ts` stays untouched. The values match the keys the
 * fork has always persisted, so existing users keep their saved settings.
 *
 * The settings *registry* entries that surface these in the UI live in
 * `$lib/fork/settings/fork-sections.ts`.
 */

export const FORK_SETTINGS_KEYS = {
	ACCENT_COLOR: 'accentColor',
	CHAT_WIDTH_STYLE: 'chatWidthStyle',
	COMMAND_PALETTE_ENABLED: 'commandPaletteEnabled',
	PRESETS_ENABLED: 'presetsEnabled',
	SKILL_SYSTEM_ENABLED: 'skillSystemEnabled',
	THEME_STYLE: 'themeStyle',
	WEB_SEARCH_ACTIVE_PROVIDER: 'webSearchActiveProvider',
	WEB_SEARCH_AUTO_DETECT: 'webSearchAutoDetect',
	WEB_SEARCH_ENABLED: 'webSearchEnabled',
	WEB_SEARCH_RESULTS_COUNT: 'webSearchResultsCount'
} as const;

export type ForkSettingsKey = (typeof FORK_SETTINGS_KEYS)[keyof typeof FORK_SETTINGS_KEYS];
