/**
 * Database-related constants (IndexedDB, Dexie).
 *
 * Centralized to ensure consistency across the app and simplify future
 * naming changes.
 */

import { STORAGE_APP_NAME } from './storage';

/** IndexedDB database name */
export const DB_NAME = STORAGE_APP_NAME;

/** IndexedDB store / table names */
export const IDXDB_TABLES = {
	conversations: 'conversations',
	messages: 'messages',
	folders: 'folders',
	skills: 'skills',
	presets: 'presets',
	searchProviders: 'searchProviders'
} as const;

/** IndexedDB store schemas */
export const IDXDB_STORE_SCHEMAS = {
	conversations: 'id, lastModified, currNode, name',
	messages: 'id, convId, type, role, timestamp, parent, children',
	folders: 'id, name, order, createdAt',
	skills: 'id, name, category, createdAt, lastUsedAt',
	presets: 'id, name, createdAt',
	searchProviders: 'id, type, name, enabled, priority'
} as const;

/** Combined Dexie stores definition — keys are table names, values are schemas */
export const IDXDB_STORES = {
	[IDXDB_TABLES.conversations]: IDXDB_STORE_SCHEMAS.conversations,
	[IDXDB_TABLES.messages]: IDXDB_STORE_SCHEMAS.messages,
	[IDXDB_TABLES.folders]: IDXDB_STORE_SCHEMAS.folders,
	[IDXDB_TABLES.skills]: IDXDB_STORE_SCHEMAS.skills,
	[IDXDB_TABLES.presets]: IDXDB_STORE_SCHEMAS.presets,
	[IDXDB_TABLES.searchProviders]: IDXDB_STORE_SCHEMAS.searchProviders
} as const;
