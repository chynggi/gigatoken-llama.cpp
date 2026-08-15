/**
 * Row types for the fork-owned IndexedDB tables declared in `fork-stores.ts`.
 *
 * `folders` has no type here on purpose — the table is retained for schema
 * compatibility only, with no code reading or writing it.
 */

import type { McpServerOverride } from '$lib/types/database';

/** Supported web search backends */
export type SearchProviderType = 'brave' | 'ddgs' | 'searxng' | 'serper' | 'tavily';

/** A `{{placeholder}}` variable declared by a skill template */
export interface SkillPlaceholder {
	name: string;
	description: string;
	defaultValue?: string;
}

/** A reusable prompt template with placeholder variables */
export interface DatabaseSkill {
	id: string;
	name: string;
	description: string;
	icon?: string;
	/** Template content with {{placeholder}} variables */
	content: string;
	/** Category for grouping (writing, coding, analysis, reasoning) */
	category?: string;
	/** Placeholder definitions for argument autocomplete */
	placeholders?: SkillPlaceholder[];
	/** Whether this is a built-in skill (non-deletable) */
	isBuiltIn?: boolean;
	createdAt: number;
	lastUsedAt?: number;
	usageCount?: number;
}

/** A configured web search backend */
export interface DatabaseSearchProvider {
	id: string;
	type: SearchProviderType;
	name: string;
	enabled: boolean;
	apiKey?: string;
	baseUrl?: string;
	config?: Record<string, string>;
	priority: number;
	createdAt: number;
}

/** A saved bundle of chat settings that can be applied in one action */
export interface DatabasePreset {
	id: string;
	name: string;
	systemMessage?: string;
	/** Sampling config keys (e.g. temperature, top_p) mapped to values */
	samplingParams?: Partial<Record<string, number | string | boolean>>;
	mcpOverrides?: McpServerOverride[];
	/** Whether auto web search is enabled for this preset */
	webSearchEnabled?: boolean;
	/** Active web search provider when this preset is applied */
	webSearchProvider?: string;
	createdAt: number;
}
