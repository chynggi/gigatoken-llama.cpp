import type { ChatMessageTimings, ChatRole, ChatMessageType } from '$lib/types/chat';
import { AttachmentType, ReasoningEffort } from '$lib/enums';

export interface McpServerOverride {
	serverId: string;
	enabled: boolean;
}

export interface DatabaseConversation {
	currNode: string | null;
	id: string;
	lastModified: number;
	name: string;
	mcpServerOverrides?: McpServerOverride[];
	thinkingEnabled?: boolean;
	reasoningEffort?: ReasoningEffort;
	forkedFromConversationId?: string;
	pinned?: boolean;
	/** Folder ID for organization */
	folderId?: string;
	/** Tags for filtering */
	tags?: string[];
	/** Whether conversation is archived */
	archived?: boolean;
}

/** Folder for organizing conversations */
export interface DatabaseFolder {
	id: string;
	name: string;
	color?: string;
	order: number;
	createdAt: number;
}

/** Reusable skill / prompt template with placeholder support */
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

export interface SkillPlaceholder {
	name: string;
	description: string;
	defaultValue?: string;
}

/**
 * Supported search provider types. Keep this in sync with SEARCH_PROVIDERS
 * in `$lib/utils/search` - only implemented providers belong here.
 */
export type SearchProviderType = 'searxng' | 'ddgs' | 'tavily' | 'brave' | 'serper';

/** Search provider configuration stored in IndexedDB */
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

/** Chat preset - saved combination of settings for one-click apply */
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

export interface DatabaseMessageExtraAudioFile {
	type: AttachmentType.AUDIO;
	name: string;
	size?: number;
	base64Data: string;
	mimeType: string;
}

export interface DatabaseMessageExtraVideoFile {
	type: AttachmentType.VIDEO;
	name: string;
	size?: number;
	base64Data: string;
	mimeType: string;
}

export interface DatabaseMessageExtraImageFile {
	type: AttachmentType.IMAGE;
	name: string;
	size?: number;
	base64Url: string;
}

/**
 * Legacy format from the old UI — pasted content was stored as "context" type
 * @deprecated Use DatabaseMessageExtraTextFile instead
 */
export interface DatabaseMessageExtraLegacyContext {
	type: AttachmentType.LEGACY_CONTEXT;
	name: string;
	size?: number;
	content: string;
}

export interface DatabaseMessageExtraPdfFile {
	type: AttachmentType.PDF;
	base64Data: string;
	name: string;
	size?: number;
	content: string;
	images?: string[];
	processedAsImages: boolean;
}

export interface DatabaseMessageExtraTextFile {
	type: AttachmentType.TEXT;
	name: string;
	size?: number;
	content: string;
}

export interface DatabaseMessageExtraMcpPrompt {
	type: AttachmentType.MCP_PROMPT;
	name: string;
	size?: number;
	serverName: string;
	promptName: string;
	content: string;
	arguments?: Record<string, string>;
}

export interface DatabaseMessageExtraMcpResource {
	type: AttachmentType.MCP_RESOURCE;
	name: string;
	size?: number;
	uri: string;
	serverName: string;
	content: string;
	mimeType?: string;
}

export type DatabaseMessageExtra =
	| DatabaseMessageExtraImageFile
	| DatabaseMessageExtraTextFile
	| DatabaseMessageExtraAudioFile
	| DatabaseMessageExtraVideoFile
	| DatabaseMessageExtraPdfFile
	| DatabaseMessageExtraMcpPrompt
	| DatabaseMessageExtraMcpResource
	| DatabaseMessageExtraLegacyContext;

export interface DatabaseMessage {
	id: string;
	convId: string;
	type: ChatMessageType;
	timestamp: number;
	role: ChatRole;
	content: string;
	parent: string | null;
	/**
	 * @deprecated - left for backward compatibility
	 */
	thinking?: string;
	/** Reasoning content produced by the model (separate from visible content) */
	reasoningContent?: string;
	/** Serialized JSON array of tool calls made by assistant messages */
	toolCalls?: string;
	/** Chat completion id streamed by the server, used to target realtime control (e.g. end reasoning) */
	completionId?: string;
	/** Tool call ID for tool result messages (role: 'tool') */
	toolCallId?: string;
	children: string[];
	extra?: DatabaseMessageExtra[];
	timings?: ChatMessageTimings;
	model?: string;
}

export type ExportedConversation = {
	conv: DatabaseConversation;
	messages: DatabaseMessage[];
};

export type ExportedConversations = ExportedConversation | ExportedConversation[];
