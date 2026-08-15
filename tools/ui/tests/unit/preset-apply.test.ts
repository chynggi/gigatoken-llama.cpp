import { SETTINGS_KEYS } from '$lib/constants';
import {
	applyMcpOverridesToServers,
	buildPresetConfigUpdates,
	collectSamplingParamsFromForm,
	deriveMcpOverridesFromServers,
	parseMcpOverridesJson,
	PRESET_SAMPLING_KEYS
} from '$lib/fork/packs/preset-apply';
import { FORK_SETTINGS_KEYS } from '$lib/fork/settings/keys';
import { describe, expect, it } from 'vitest';

const SERVERS = JSON.stringify([
	{ enabled: false, id: 's1', url: 'https://one.example' },
	{ enabled: true, id: 's2', url: 'https://two.example' }
]);

describe('buildPresetConfigUpdates', () => {
	it('emits system message, sampling config keys, merged mcpServers, and web search fields', () => {
		const updates = buildPresetConfigUpdates(
			{
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
			},
			SERVERS
		);

		const byKey = Object.fromEntries(updates.map((u) => [u.key, u.value]));
		expect(byKey[SETTINGS_KEYS.SYSTEM_MESSAGE]).toBe('Be concise');
		expect(byKey[SETTINGS_KEYS.TEMPERATURE]).toBe(0.2);
		expect(byKey[SETTINGS_KEYS.TOP_P]).toBe(0.9);
		expect(byKey['TEMPERATURE']).toBeUndefined();
		expect(JSON.parse(byKey[SETTINGS_KEYS.MCP_SERVERS] as string)).toEqual([
			{ enabled: true, id: 's1', url: 'https://one.example' },
			{ enabled: true, id: 's2', url: 'https://two.example' }
		]);
		expect(byKey[FORK_SETTINGS_KEYS.WEB_SEARCH_ENABLED]).toBe(true);
		expect(byKey[FORK_SETTINGS_KEYS.WEB_SEARCH_ACTIVE_PROVIDER]).toBe('prov-1');
	});

	it('omits the mcpServers update when no current server list is supplied', () => {
		const updates = buildPresetConfigUpdates({
			mcpOverrides: [{ serverId: 's1', enabled: true }]
		});

		expect(updates.some((u) => u.key === SETTINGS_KEYS.MCP_SERVERS)).toBe(false);
	});

	it('skips empty webSearchProvider', () => {
		const updates = buildPresetConfigUpdates({
			webSearchProvider: ''
		});
		expect(updates.some((u) => u.key === FORK_SETTINGS_KEYS.WEB_SEARCH_ACTIVE_PROVIDER)).toBe(
			false
		);
	});
});

describe('applyMcpOverridesToServers', () => {
	it('rewrites only the servers named by an override and leaves other fields intact', () => {
		const merged = applyMcpOverridesToServers(SERVERS, [{ serverId: 's2', enabled: false }]);

		expect(JSON.parse(merged as string)).toEqual([
			{ enabled: false, id: 's1', url: 'https://one.example' },
			{ enabled: false, id: 's2', url: 'https://two.example' }
		]);
	});

	it('returns null when nothing matched or input is unusable', () => {
		expect(applyMcpOverridesToServers(SERVERS, [{ serverId: 'nope', enabled: true }])).toBeNull();
		expect(applyMcpOverridesToServers('not-json', [{ serverId: 's1', enabled: true }])).toBeNull();
		expect(applyMcpOverridesToServers('{}', [{ serverId: 's1', enabled: true }])).toBeNull();
	});
});

describe('deriveMcpOverridesFromServers', () => {
	it('reads each server id with its enabled flag', () => {
		expect(deriveMcpOverridesFromServers(SERVERS)).toEqual([
			{ enabled: false, serverId: 's1' },
			{ enabled: true, serverId: 's2' }
		]);
	});

	it('returns undefined for unusable or empty input', () => {
		expect(deriveMcpOverridesFromServers('not-json')).toBeUndefined();
		expect(deriveMcpOverridesFromServers('{}')).toBeUndefined();
		expect(deriveMcpOverridesFromServers('[]')).toBeUndefined();
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
			expect(key).toBe(key.toLowerCase());
		}
		expect(PRESET_SAMPLING_KEYS).toContain('temperature');
		expect(PRESET_SAMPLING_KEYS).not.toContain('TEMPERATURE');
	});
});

describe('DatabasePreset surface', () => {
	it('has no dead icon field in the shipped type source', async () => {
		const { readFileSync } = await import('node:fs');
		const { join } = await import('node:path');
		const src = readFileSync(join(process.cwd(), 'src/lib/fork/db/types.ts'), 'utf8');
		const presetBlock = src.slice(src.indexOf('export interface DatabasePreset'));

		expect(presetBlock).toContain('systemMessage');
		expect(presetBlock).toContain('samplingParams');
		expect(presetBlock).toContain('mcpOverrides');
		expect(presetBlock).toContain('webSearchEnabled');
		expect(presetBlock).not.toMatch(/\bicon\??:/);
	});
});
