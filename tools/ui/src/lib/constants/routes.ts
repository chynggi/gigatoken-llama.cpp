export const NEW_CHAT_PARAM = 'new_chat';

/** Settings section slugs — used for routes and navigation. */
export const SETTINGS_SECTION_SLUGS = {
	GENERAL: 'general',
	DISPLAY: 'display',
	SAMPLING: 'sampling',
	PENALTIES: 'penalties',
	AGENTIC: 'agentic',
	DEVELOPER: 'developer',
	TOOLS: 'tools',
	IMPORT_EXPORT: 'import-export',
	MCP: 'mcp',
	PACKS: 'packs'
} as const;

export const ROUTES = {
	/** Root - start of the app. */
	START: '#/',
	/** New chat - root with new chat query param. */
	NEW_CHAT: `?${NEW_CHAT_PARAM}=true#/`,
	/** Chat base - for dynamic chat URLs use RouterService. */
	CHAT: '#/chat',
	/** MCP servers. */
	MCP_SERVERS: '#/mcp-servers',
	/** Settings base - for dynamic settings URLs use RouterService. */
	SETTINGS: '#/settings',
	/** Search - mobile-only full-page conversation search. */
	SEARCH: '#/search',
	/** Packs management pages */
	SKILLS: '#/skills',
	PRESETS: '#/presets',
	SEARCH_PROVIDERS: '#/search-providers'
} as const;

/** Custom event to fill the chat compose input (command palette skills, etc.) */
export const COMPOSE_CHAT_INPUT_EVENT = 'compose-chat-input';
