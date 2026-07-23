import { describe, it, expect } from 'vitest';
import {
	buildPresetConfigUpdates,
	parseMcpOverridesJson,
	collectSamplingParamsFromForm,
	PRESET_SAMPLING_KEYS
} from '$lib/utils/preset-apply';
import { SETTINGS_KEYS } from '$lib/constants';

describe('buildPresetConfigUpdates', () => {
	it('emits system message, sampling config keys, mcp JSON, and web search fields', () => {
		const updates = buildPresetConfigUpdates({
			systemMessage: 'Be concise',
			samplingParams: {
				[SETTINGS_KEYS.TEMPERATURE]: 0.2,
				[SETTINGS_KEYS.TOP_P]: 0.9,
				// bogus enum-name style key must be ignored
				TEMPERATURE: 1
			},
			mcpOverrides: [{ serverId: 's1', enabled: true }],
			webSearchEnabled: true,
			webSearchProvider: 'prov-1'
		});

		const byKey = Object.fromEntries(updates.map((u) => [u.key, u.value]));
		expect(byKey[SETTINGS_KEYS.SYSTEM_MESSAGE]).toBe('Be concise');
		expect(byKey[SETTINGS_KEYS.TEMPERATURE]).toBe(0.2);
		expect(byKey[SETTINGS_KEYS.TOP_P]).toBe(0.9);
		expect(byKey['TEMPERATURE']).toBeUndefined();
		expect(byKey[SETTINGS_KEYS.MCP_DEFAULT_SERVER_OVERRIDES]).toBe(
			JSON.stringify([{ serverId: 's1', enabled: true }])
		);
		expect(byKey[SETTINGS_KEYS.WEB_SEARCH_ENABLED]).toBe(true);
		expect(byKey[SETTINGS_KEYS.WEB_SEARCH_ACTIVE_PROVIDER]).toBe('prov-1');
	});

	it('skips empty webSearchProvider', () => {
		const updates = buildPresetConfigUpdates({
			webSearchProvider: ''
		});
		expect(updates.some((u) => u.key === SETTINGS_KEYS.WEB_SEARCH_ACTIVE_PROVIDER)).toBe(false);
	});
});

describe('parseMcpOverridesJson', () => {
	it('parses valid override arrays', () => {
		expect(parseMcpOverridesJson('[{"serverId":"a","enabled":false}]')).toEqual([
			{ serverId: 'a', enabled: false }
		]);
	});

	it('returns undefined for empty and null for invalid', () => {
		expect(parseMcpOverridesJson('')).toBeUndefined();
		expect(parseMcpOverridesJson('{}')).toBeNull();
		expect(parseMcpOverridesJson('not-json')).toBeNull();
		expect(parseMcpOverridesJson('[{"serverId":1,"enabled":true}]')).toBeNull();
	});
});

describe('collectSamplingParamsFromForm', () => {
	it('collects only numeric known sampling keys', () => {
		const params = collectSamplingParamsFromForm({
			[SETTINGS_KEYS.TEMPERATURE]: '0.7',
			[SETTINGS_KEYS.TOP_K]: '',
			[SETTINGS_KEYS.MAX_TOKENS]: '512'
		});
		expect(params).toEqual({
			[SETTINGS_KEYS.TEMPERATURE]: 0.7,
			[SETTINGS_KEYS.MAX_TOKENS]: 512
		});
	});

	it('returns undefined when nothing set', () => {
		expect(collectSamplingParamsFromForm({})).toBeUndefined();
	});

	it('exposes PRESET_SAMPLING_KEYS as real config property paths', () => {
		const values = new Set(Object.values(SETTINGS_KEYS));
		for (const key of PRESET_SAMPLING_KEYS) {
			expect(values.has(key)).toBe(true);
			// not enum names
			expect(key.includes('_') || key === 'temperature' || key.startsWith('top') || key.startsWith('min') || key.startsWith('max') || key.startsWith('repeat')).toBe(
				true
			);
			expect(key === key.toLowerCase() || key.includes('_')).toBe(true);
		}
		expect(PRESET_SAMPLING_KEYS).toContain('temperature');
		expect(PRESET_SAMPLING_KEYS).not.toContain('TEMPERATURE');
	});
});

describe('DatabasePreset surface', () => {
	it('has no dead icon field in the shipped type source', async () => {
		const { readFileSync } = await import('node:fs');
		const { join } = await import('node:path');
		const src = readFileSync(join(process.cwd(), 'src/lib/types/database.d.ts'), 'utf8');
		const presetBlock = src.slice(src.indexOf('export interface DatabasePreset'), src.indexOf('export interface DatabaseMessageExtraAudioFile'));
		expect(presetBlock).toContain('systemMessage');
		expect(presetBlock).toContain('samplingParams');
		expect(presetBlock).toContain('mcpOverrides');
		expect(presetBlock).toContain('webSearchEnabled');
		expect(presetBlock).not.toMatch(/\bicon\??:/);
	});
});

