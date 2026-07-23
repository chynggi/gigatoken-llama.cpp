import { describe, expect, it } from 'vitest';
import {
	RECOMMENDED_MCP_SERVER_IDS,
	RECOMMENDED_MCP_SERVERS
} from '$lib/constants/recommended-mcp-servers';
import { MCP_SERVER_ID_PREFIX } from '$lib/constants/mcp';

describe('RECOMMENDED_MCP_SERVERS', () => {
	it('lists at least one entry and uses stable, unique ids', () => {
		expect(RECOMMENDED_MCP_SERVERS.length).toBeGreaterThan(0);

		const ids = RECOMMENDED_MCP_SERVERS.map((server) => server.id);
		expect(new Set(ids).size).toBe(ids.length);

		for (const id of ids) {
			expect(id).toMatch(/^[a-z0-9-]+$/);
			expect(id.toLowerCase()).not.toContain(MCP_SERVER_ID_PREFIX.toLowerCase());
		}
	});

	it('requires a name, description and url for every entry', () => {
		for (const server of RECOMMENDED_MCP_SERVERS) {
			expect(server.name?.trim().length ?? 0).toBeGreaterThan(0);
			expect(server.description.trim().length).toBeGreaterThan(0);
			expect(server.url.trim().length).toBeGreaterThan(0);
			expect(() => new URL(server.url)).not.toThrow();
		}
	});
});

describe('RECOMMENDED_MCP_SERVER_IDS', () => {
	it('matches the ids declared in RECOMMENDED_MCP_SERVERS', () => {
		expect(RECOMMENDED_MCP_SERVER_IDS.size).toBe(RECOMMENDED_MCP_SERVERS.length);

		for (const server of RECOMMENDED_MCP_SERVERS) {
			expect(RECOMMENDED_MCP_SERVER_IDS.has(server.id)).toBe(true);
		}
	});
});
