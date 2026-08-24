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

import { SETTINGS_SECTION_SLUGS } from '$lib/constants/settings.constants';
import { FORK_SETTINGS_KEYS } from '$lib/fork/settings/keys';
import { SettingsFieldType } from '$lib/enums/settings.enums';
import type { SettingsEntry, SettingsSectionEntry, SettingsSectionTitle } from '$lib/types';
import { Circle, Columns2, Package, Palette } from '@lucide/svelte';

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
	{ value: 'default', label: 'Default', icon: Circle },
	{ value: 'blue', label: 'Blue', icon: Circle },
	{ value: 'green', label: 'Green', icon: Circle },
	{ value: 'purple', label: 'Purple', icon: Circle },
	{ value: 'orange', label: 'Orange', icon: Circle },
	{ value: 'pink', label: 'Pink', icon: Circle },
	{ value: 'red', label: 'Red', icon: Circle }
];

const THEME_STYLE_OPTIONS = [
	{ value: 'default', label: 'Default', icon: Palette },
	{ value: 'tokyo-night', label: 'Tokyo Night', icon: Palette },
	{ value: 'nord', label: 'Nord', icon: Palette },
	{ value: 'dracula', label: 'Dracula', icon: Palette },
	{ value: 'gruvbox', label: 'Gruvbox', icon: Palette },
	{ value: 'synthwave', label: "Synthwave '84", icon: Palette },
	{ value: 'soft', label: 'Soft (Gradio)', icon: Palette },
	{ value: 'monochrome', label: 'Monochrome (Gradio)', icon: Palette }
];

const CHAT_WIDTH_STYLE_OPTIONS = [
	{ value: 'normal', label: 'Normal (Centered)', icon: Columns2 },
	{ value: 'wide', label: 'Wide', icon: Columns2 },
	{ value: 'full', label: 'Full Width (Fluid)', icon: Columns2 }
];

/**
 * Theme controls, appended to upstream's Display section.
 *
 * `ThemeEffects.svelte` reads these and writes the matching CSS custom
 * properties and root classes that `fork.css` styles against.
 */
export const FORK_DISPLAY_SETTINGS: SettingsEntry[] = [
	{
		key: FORK_SETTINGS_KEYS.ACCENT_COLOR,
		label: 'Accent color',
		help: 'Choose a custom accent color used for focus glows, message bubbles, and AI activity highlights.',
		defaultValue: 'default',
		type: SettingsFieldType.SELECT,
		options: ACCENT_COLOR_OPTIONS
	},
	{
		key: FORK_SETTINGS_KEYS.THEME_STYLE,
		label: 'Theme style variation',
		help: 'Choose a specific theme style variation to customize color schemes and syntax highlighting.',
		defaultValue: 'default',
		type: SettingsFieldType.SELECT,
		options: THEME_STYLE_OPTIONS
	},
	{
		key: FORK_SETTINGS_KEYS.CHAT_WIDTH_STYLE,
		label: 'Chat layout width',
		help: 'Choose how wide the chat message list should expand. Full Width fills the entire window space except the sidebar.',
		defaultValue: 'normal',
		type: SettingsFieldType.SELECT,
		options: CHAT_WIDTH_STYLE_OPTIONS
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
		title: FORK_PACKS_TITLE,
		slug: FORK_PACKS_SLUG,
		icon: Package,
		settings: [
			{
				key: FORK_SETTINGS_KEYS.SKILL_SYSTEM_ENABLED,
				label: 'Skill system',
				help: 'Enable reusable prompt templates with slash commands (/summarize, /translate, etc.). Manage skills at #/skills.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				isExperimental: true
			},
			{
				key: FORK_SETTINGS_KEYS.WEB_SEARCH_ENABLED,
				label: 'Web search',
				help: 'Inject web search results into chat context on send. Configure providers at #/search-providers. When auto-detect is off, every message is searched; when on, only messages ending with ?.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				isExperimental: true
			},
			{
				key: FORK_SETTINGS_KEYS.WEB_SEARCH_AUTO_DETECT,
				label: 'Auto-detect search queries',
				help: 'When enabled, only search for messages that end with a question mark. When disabled, search every message while web search is on.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				isExperimental: true
			},
			{
				key: FORK_SETTINGS_KEYS.WEB_SEARCH_RESULTS_COUNT,
				label: 'Search results count',
				help: 'Number of search results to inject into chat context (1-10).',
				defaultValue: 5,
				type: SettingsFieldType.INPUT,
				isPositiveInteger: true,
				isExperimental: true
			},
			{
				key: FORK_SETTINGS_KEYS.WEB_SEARCH_ACTIVE_PROVIDER,
				label: 'Active search provider',
				help: 'ID of the default search provider. Leave empty to use the first enabled provider. Manage providers at #/search-providers.',
				defaultValue: '',
				type: SettingsFieldType.INPUT,
				isExperimental: true
			},
			{
				key: FORK_SETTINGS_KEYS.PRESETS_ENABLED,
				label: 'Chat presets',
				help: 'Enable saved presets (system message + sampling + MCP + web search). Apply from the chat form, command palette, or #/presets.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				isExperimental: true
			},
			{
				key: FORK_SETTINGS_KEYS.COMMAND_PALETTE_ENABLED,
				label: 'Command palette',
				help: 'Enable Cmd/Ctrl+K command palette for quick actions: search conversations, run skills, apply presets, and more.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				isExperimental: true
			}
		]
	}
};
