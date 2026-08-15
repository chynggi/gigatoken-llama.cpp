/**
 * Fork-owned IndexedDB tables.
 *
 * Declared separately from upstream's `IDXDB_STORES` so that upstream's
 * `database.service.ts` carries a single spread hook instead of a diverged
 * table list.
 *
 * See `docs/superpowers/specs/2026-08-15-ui-upstream-realign-design.md`.
 */

/** Fork table names */
export const FORK_TABLES = {
	folders: 'folders',
	presets: 'presets',
	searchProviders: 'searchProviders',
	skills: 'skills'
} as const;

/**
 * Fork table schemas.
 *
 * `folders` is kept declared but unused. The Folders/Tags feature was dropped
 * during the upstream realignment, and lowering a Dexie schema version throws
 * `VersionError` against databases that already reached version 2. Leaving the
 * store declared costs nothing and preserves existing user data in case the
 * feature is restored.
 */
export const FORK_STORE_SCHEMAS = {
	folders: 'id, name, order, createdAt',
	presets: 'id, name, createdAt',
	searchProviders: 'id, type, name, enabled, priority',
	skills: 'id, name, category, createdAt, lastUsedAt'
} as const;

/** Combined fork stores definition — spread into Dexie's `version(2)` */
export const FORK_STORES = {
	[FORK_TABLES.folders]: FORK_STORE_SCHEMAS.folders,
	[FORK_TABLES.presets]: FORK_STORE_SCHEMAS.presets,
	[FORK_TABLES.searchProviders]: FORK_STORE_SCHEMAS.searchProviders,
	[FORK_TABLES.skills]: FORK_STORE_SCHEMAS.skills
} as const;
