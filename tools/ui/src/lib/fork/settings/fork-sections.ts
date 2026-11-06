/**
 * Settings registry entries owned by the fork.
 *
 * Upstream's `settings.constants.ts` spreads these in, which keeps the
 * fork's footprint there to three lines. `SETTING_CONFIG_DEFAULT` and
 * `SETTINGS_CHAT_SECTIONS` are both derived from the registry, so defaults and
 * the settings UI pick these up with no further wiring.
 *
 * The key strings themselves live in `keys.ts`.
 */

import { Circle, Columns2, Package, Palette } from '@lucide/svelte';
import { SettingsFieldType } from '$lib/enums/settings.enums';
import { FORK_SETTINGS_KEYS } from '$lib/fork/settings/keys';
import type { SettingsEntry, SettingsSectionEntry, SettingsSectionTitle } from '$lib/types';

/** Slug for the fork's own settings section. */
export const FORK_PACKS_SLUG = 'packs';

/**
 * `SettingsSectionTitle` is a union over upstream's own section titles, so the
 * fork's section name is asserted rather than added to that constant — the
 * point of this module is to leave upstream's registry file alone.
 */
const FORK_PACKS_TITLE = 'Packs' as SettingsSectionTitle;
/**
 * `SettingsEntry.options` requires an icon (the derived UI type treats it as
 * optional). One representative icon per group keeps the select rows uniform.
 */
const ACCENT_COLOR_OPTIONS = [
	{ icon: Circle, label: 'Default', value: 'default' },
	{ icon: Circle, label: 'Blue', value: 'blue' },
	{ icon: Circle, label: 'Green', value: 'green' },
	{ icon: Circle, label: 'Purple', value: 'purple' },
	{ icon: Circle, label: 'Orange', value: 'orange' },
	{ icon: Circle, label: 'Pink', value: 'pink' },
	{ icon: Circle, label: 'Red', value: 'red' }
];
const THEME_STYLE_OPTIONS = [
	{ icon: Palette, label: 'Default', value: 'default' },
	{ icon: Palette, label: 'Tokyo Night', value: 'tokyo-night' },
	{ icon: Palette, label: 'Nord', value: 'nord' },
	{ icon: Palette, label: 'Dracula', value: 'dracula' },
	{ icon: Palette, label: 'Gruvbox', value: 'gruvbox' },
	{ icon: Palette, label: "Synthwave '84", value: 'synthwave' },
	{ icon: Palette, label: 'Soft (Gradio)', value: 'soft' },
	{ icon: Palette, label: 'Monochrome (Gradio)', value: 'monochrome' }
];
const CHAT_WIDTH_STYLE_OPTIONS = [
	{ icon: Columns2, label: 'Normal (Centered)', value: 'normal' },
	{ icon: Columns2, label: 'Wide', value: 'wide' },
	{ icon: Columns2, label: 'Full Width (Fluid)', value: 'full' }
];

/**
 * Theme controls, appended to upstream's Display section.
 *
 * `ThemeEffects.svelte` reads these and writes the matching CSS custom
 * properties and root classes that `fork.css` styles against.
 */
export const FORK_DISPLAY_SETTINGS: SettingsEntry[] = [
	{
		defaultValue: 'default',
		help: 'Choose a custom accent color used for focus glows, message bubbles, and AI activity highlights.',
		key: FORK_SETTINGS_KEYS.ACCENT_COLOR,
		label: 'Accent color',
		options: ACCENT_COLOR_OPTIONS,
		type: SettingsFieldType.SELECT
	},
	{
		defaultValue: 'default',
		help: 'Choose a specific theme style variation to customize color schemes and syntax highlighting.',
		key: FORK_SETTINGS_KEYS.THEME_STYLE,
		label: 'Theme style variation',
		options: THEME_STYLE_OPTIONS,
		type: SettingsFieldType.SELECT
	},
	{
		defaultValue: 'normal',
		help: 'Choose how wide the chat message list should expand. Full Width fills the entire window space except the sidebar.',
		key: FORK_SETTINGS_KEYS.CHAT_WIDTH_STYLE,
		label: 'Chat layout width',
		options: CHAT_WIDTH_STYLE_OPTIONS,
		type: SettingsFieldType.SELECT
	}
];

/**
 * The fork's own settings section, spread into upstream's registry.
 *
 * Every entry is opt-in and marked experimental: the features stay dormant
 * until switched on here.
 */
export const FORK_SETTINGS_REGISTRY: Record<string, SettingsSectionEntry> = {
	[FORK_PACKS_SLUG]: {
		icon: Package,
		settings: [
			{
				defaultValue: false,
				help: 'Enable reusable prompt templates with slash commands (/summarize, /translate, etc.). Manage skills at #/skills.',
				isExperimental: true,
				key: FORK_SETTINGS_KEYS.SKILL_SYSTEM_ENABLED,
				label: 'Skill system',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Inject web search results into chat context on send. Configure providers at #/search-providers. When auto-detect is off, every message is searched; when on, only messages ending with ?.',
				isExperimental: true,
				key: FORK_SETTINGS_KEYS.WEB_SEARCH_ENABLED,
				label: 'Web search',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'When enabled, only search for messages that end with a question mark. When disabled, search every message while web search is on.',
				isExperimental: true,
				key: FORK_SETTINGS_KEYS.WEB_SEARCH_AUTO_DETECT,
				label: 'Auto-detect search queries',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: 5,
				help: 'Number of search results to inject into chat context (1-10).',
				isExperimental: true,
				isPositiveInteger: true,
				key: FORK_SETTINGS_KEYS.WEB_SEARCH_RESULTS_COUNT,
				label: 'Search results count',
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: '',
				help: 'ID of the default search provider. Leave empty to use the first enabled provider. Manage providers at #/search-providers.',
				isExperimental: true,
				key: FORK_SETTINGS_KEYS.WEB_SEARCH_ACTIVE_PROVIDER,
				label: 'Active search provider',
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: false,
				help: 'Enable saved presets (system message + sampling + MCP + web search). Apply from the chat form, command palette, or #/presets.',
				isExperimental: true,
				key: FORK_SETTINGS_KEYS.PRESETS_ENABLED,
				label: 'Chat presets',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Enable Cmd/Ctrl+K command palette for quick actions: search conversations, run skills, apply presets, and more.',
				isExperimental: true,
				key: FORK_SETTINGS_KEYS.COMMAND_PALETTE_ENABLED,
				label: 'Command palette',
				type: SettingsFieldType.CHECKBOX
			}
		],
		slug: FORK_PACKS_SLUG,
		title: FORK_PACKS_TITLE
	}
};
