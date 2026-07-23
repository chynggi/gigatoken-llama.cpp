/**
 * Pure preset application: maps DatabasePreset fields onto config updates.
 * Store layer only persists the resulting key/value pairs.
 */

import { SETTINGS_KEYS } from '$lib/constants';
import type { SettingsConfigType } from '$lib/types/settings';

/** Sampling keys exposed in the preset editor (config property paths). */
export const PRESET_SAMPLING_KEYS = [
	SETTINGS_KEYS.TEMPERATURE,
	SETTINGS_KEYS.TOP_P,
	SETTINGS_KEYS.TOP_K,
	SETTINGS_KEYS.MIN_P,
	SETTINGS_KEYS.MAX_TOKENS,
	SETTINGS_KEYS.REPEAT_PENALTY
] as const;

export type PresetSamplingKey = (typeof PRESET_SAMPLING_KEYS)[number];

export type PresetConfigUpdate = {
	key: keyof SettingsConfigType;
	value: SettingsConfigType[keyof SettingsConfigType];
};

const knownConfigKeys = new Set<string>(Object.values(SETTINGS_KEYS));

/**
 * Build ordered config updates from a preset. Only emits keys that exist on
 * SettingsConfigType (via SETTINGS_KEYS values).
 */
export function buildPresetConfigUpdates(
	preset: Partial<
		Pick<
			DatabasePreset,
			| 'systemMessage'
			| 'samplingParams'
			| 'mcpOverrides'
			| 'webSearchEnabled'
			| 'webSearchProvider'
		>
	>
): PresetConfigUpdate[] {
	const updates: PresetConfigUpdate[] = [];

	if (preset.systemMessage !== undefined) {
		updates.push({
			key: SETTINGS_KEYS.SYSTEM_MESSAGE as keyof SettingsConfigType,
			value: preset.systemMessage
		});
	}

	if (preset.samplingParams) {
		for (const [key, value] of Object.entries(preset.samplingParams)) {
			if (!knownConfigKeys.has(key) || value === undefined) continue;
			updates.push({
				key: key as keyof SettingsConfigType,
				value: value as SettingsConfigType[keyof SettingsConfigType]
			});
		}
	}

	if (preset.mcpOverrides) {
		updates.push({
			key: SETTINGS_KEYS.MCP_DEFAULT_SERVER_OVERRIDES as keyof SettingsConfigType,
			value: JSON.stringify(preset.mcpOverrides)
		});
	}

	if (preset.webSearchEnabled !== undefined) {
		updates.push({
			key: SETTINGS_KEYS.WEB_SEARCH_ENABLED as keyof SettingsConfigType,
			value: preset.webSearchEnabled
		});
	}

	if (preset.webSearchProvider !== undefined && preset.webSearchProvider !== '') {
		updates.push({
			key: SETTINGS_KEYS.WEB_SEARCH_ACTIVE_PROVIDER as keyof SettingsConfigType,
			value: preset.webSearchProvider
		});
	}

	return updates;
}

/** Parse MCP overrides JSON from the editor; returns null on invalid input. */
export function parseMcpOverridesJson(
	raw: string
): DatabasePreset['mcpOverrides'] | null | undefined {
	const trimmed = raw.trim();
	if (!trimmed) return undefined;
	try {
		const parsed = JSON.parse(trimmed);
		if (!Array.isArray(parsed)) return null;
		const ok = parsed.every(
			(o) =>
				typeof o === 'object' &&
				o !== null &&
				typeof (o as { serverId?: unknown }).serverId === 'string' &&
				typeof (o as { enabled?: unknown }).enabled === 'boolean'
		);
		if (!ok) return null;
		return parsed as DatabasePreset['mcpOverrides'];
	} catch {
		return null;
	}
}

/** Collect sampling params from editor string fields (empty = omit). */
export function collectSamplingParamsFromForm(
	fields: Partial<Record<PresetSamplingKey, string>>
): DatabasePreset['samplingParams'] | undefined {
	const out: Record<string, number> = {};
	for (const key of PRESET_SAMPLING_KEYS) {
		const raw = fields[key];
		if (raw === undefined || raw.trim() === '') continue;
		const n = Number(raw);
		if (Number.isNaN(n)) continue;
		out[key] = n;
	}
	return Object.keys(out).length > 0 ? out : undefined;
}
