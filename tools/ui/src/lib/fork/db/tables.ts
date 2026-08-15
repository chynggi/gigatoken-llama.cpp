/**
 * Typed Dexie accessors for the fork-owned tables.
 *
 * Upstream's `LlamaUiDatabase` declares `EntityTable` properties only for its
 * own tables. Rather than widening that class (which would grow the fork's
 * footprint inside an upstream file), fork tables are reached through Dexie's
 * generic `table()` and typed here, so `forkDb` is referenced in exactly one
 * place.
 */

import { FORK_TABLES } from '$lib/fork/db/fork-stores';
import type { DatabasePreset, DatabaseSearchProvider, DatabaseSkill } from '$lib/fork/db/types';
import { forkDb } from '$lib/services/database.service';
import type { Table } from 'dexie';

export const presetsTable = forkDb.table(FORK_TABLES.presets) as Table<DatabasePreset, string>;

export const searchProvidersTable = forkDb.table(FORK_TABLES.searchProviders) as Table<
	DatabaseSearchProvider,
	string
>;

export const skillsTable = forkDb.table(FORK_TABLES.skills) as Table<DatabaseSkill, string>;
