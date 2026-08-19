/**
 * Pure preset application: maps DatabasePreset fields onto config updates.
 * Store layer only persists the resulting key/value pairs.
 */

import { SETTINGS_KEYS } from '$lib/constants';
import type { DatabasePreset } from '$lib/fork/db/types';
import { FORK_SETTINGS_KEYS } from '$lib/fork/settings/keys';
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
 * Merge a preset's MCP overrides onto the `mcpServers` config entry.
 *
 * Upstream made `mcpServers[i].enabled` the single source of truth for
 * new-chat defaults (see the `mcp-default-overrides-merge-v1` migration), so a
 * preset applies its overrides by rewriting that list rather than by writing a
 * separate defaults key.
 *
 * @param rawServers - Current `mcpServers` config value (a JSON array string)
 * @param overrides - Per-server enabled flags carried by the preset
 * @returns The updated JSON string, or `null` when `rawServers` is unusable or
 *   no listed server matched an override
 */
export function applyMcpOverridesToServers(
	rawServers: string,
	overrides: NonNullable<DatabasePreset['mcpOverrides']>
): string | null {
	let servers: unknown;

	try {
		servers = JSON.parse(rawServers);
	} catch {
		return null;
	}

	if (!Array.isArray(servers)) return null;

	const enabledById = new Map(overrides.map((o) => [o.serverId, o.enabled]));
	let touched = false;
	const next = servers.map((server) => {
		if (typeof server !== 'object' || server === null) return server;

		const id = (server as { id?: unknown }).id;

		if (typeof id !== 'string' || !enabledById.has(id)) return server;

		touched = true;

		return { ...(server as object), enabled: enabledById.get(id) };
	});

	return touched ? JSON.stringify(next) : null;
}

/**
 * Read the current per-server enabled flags out of the `mcpServers` config
 * entry, in the shape a preset stores them. Used to pre-fill the preset editor.
 *
 * @param rawServers - Current `mcpServers` config value (a JSON array string)
 * @returns The override list, or `undefined` when `rawServers` is unusable
 */
export function deriveMcpOverridesFromServers(
	rawServers: string
): DatabasePreset['mcpOverrides'] | undefined {
	let servers: unknown;

	try {
		servers = JSON.parse(rawServers);
	} catch {
		return undefined;
	}

	if (!Array.isArray(servers)) return undefined;

	const overrides = servers.flatMap((server) => {
		if (typeof server !== 'object' || server === null) return [];

		const { enabled, id } = server as { enabled?: unknown; id?: unknown };

		if (typeof id !== 'string') return [];

		return [{ enabled: Boolean(enabled), serverId: id }];
	});

	return overrides.length > 0 ? overrides : undefined;
}

/**
 * Build ordered config updates from a preset. Only emits keys that exist on
 * SettingsConfigType (via SETTINGS_KEYS values).
 *
 * @param preset - Preset fields to apply
 * @param currentMcpServers - Current `mcpServers` config value. Required for
 *   `mcpOverrides` to produce an update; omit it when the caller has no MCP
 *   state to merge onto.
 */
export function buildPresetConfigUpdates(
	preset: Partial<
		Pick<
			DatabasePreset,
			'systemMessage' | 'samplingParams' | 'mcpOverrides' | 'webSearchEnabled' | 'webSearchProvider'
		>
	>,
	currentMcpServers?: string
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

	if (preset.mcpOverrides && currentMcpServers !== undefined) {
		const merged = applyMcpOverridesToServers(currentMcpServers, preset.mcpOverrides);

		if (merged !== null) {
			updates.push({
				key: SETTINGS_KEYS.MCP_SERVERS as keyof SettingsConfigType,
				value: merged
			});
		}
	}

	if (preset.webSearchEnabled !== undefined) {
		updates.push({
			key: FORK_SETTINGS_KEYS.WEB_SEARCH_ENABLED as keyof SettingsConfigType,
			value: preset.webSearchEnabled
		});
	}

	if (preset.webSearchProvider !== undefined && preset.webSearchProvider !== '') {
		updates.push({
			key: FORK_SETTINGS_KEYS.WEB_SEARCH_ACTIVE_PROVIDER as keyof SettingsConfigType,
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
