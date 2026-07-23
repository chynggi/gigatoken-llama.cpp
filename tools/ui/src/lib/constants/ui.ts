import { Search, Settings, SquarePen } from '@lucide/svelte';
import McpLogo from '$lib/components/app/mcp/McpLogo.svelte';
import type { Component } from 'svelte';
import { ROUTES } from './routes';

export interface AccentColorDefinition {
	light: string;
	dark: string;
	glowLight: string;
	glowDark: string;
}

export const ACCENT_COLORS: Record<string, AccentColorDefinition> = {
	default: {
		light: 'oklch(0.93 0.04 220)',
		dark: 'oklch(0.4 0.2 220)',
		glowLight: 'oklch(0.6 0.22 220)',
		glowDark: 'oklch(0.85 0.22 220)'
	},
	blue: {
		light: 'oklch(0.93 0.04 250)',
		dark: 'oklch(0.3 0.14 250)',
		glowLight: 'oklch(0.6 0.18 250)',
		glowDark: 'oklch(0.85 0.14 250)'
	},
	green: {
		light: 'oklch(0.94 0.04 145)',
		dark: 'oklch(0.32 0.14 145)',
		glowLight: 'oklch(0.65 0.18 145)',
		glowDark: 'oklch(0.85 0.14 145)'
	},
	purple: {
		light: 'oklch(0.92 0.05 290)',
		dark: 'oklch(0.3 0.15 290)',
		glowLight: 'oklch(0.58 0.2 290)',
		glowDark: 'oklch(0.82 0.15 290)'
	},
	orange: {
		light: 'oklch(0.93 0.05 55)',
		dark: 'oklch(0.32 0.14 55)',
		glowLight: 'oklch(0.65 0.18 55)',
		glowDark: 'oklch(0.85 0.14 55)'
	},
	pink: {
		light: 'oklch(0.93 0.05 350)',
		dark: 'oklch(0.32 0.14 350)',
		glowLight: 'oklch(0.65 0.18 350)',
		glowDark: 'oklch(0.85 0.14 350)'
	},
	red: {
		light: 'oklch(0.92 0.05 25)',
		dark: 'oklch(0.3 0.14 25)',
		glowLight: 'oklch(0.6 0.18 25)',
		glowDark: 'oklch(0.82 0.14 25)'
	}
};

export const FORK_TREE_DEPTH_PADDING = 8;
export const SYSTEM_MESSAGE_PLACEHOLDER = 'System message';

export const ICON_STRIP_TRANSITION_DURATION = 150;
export const ICON_STRIP_TRANSITION_DELAY_MULTIPLIER = 50;

/** Max height for tool-result code blocks (json / source / diff / streaming code). */
export const MAX_HEIGHT_CODE_BLOCK = '22rem';

export interface DesktopIconStripItem {
	icon: Component;
	tooltip: string;
	route?: string;
	activeRouteId?: string;
	activeRoutePrefix?: string;
	activeUrlIncludes?: string;
	keys?: string[];
}

export const SIDEBAR_ACTIONS_ITEMS: DesktopIconStripItem[] = [
	{ icon: SquarePen, tooltip: 'New chat', route: ROUTES.NEW_CHAT, keys: ['shift', 'cmd', 'o'] },
	{ icon: Search, tooltip: 'Search', keys: ['cmd', 'k'] },
	{
		icon: McpLogo,
		tooltip: 'MCP Servers',
		route: ROUTES.MCP_SERVERS,
		activeRouteId: '/mcp-servers'
	},
	{
		icon: Settings,
		tooltip: 'Settings',
		route: `${ROUTES.SETTINGS}/general`,
		activeUrlIncludes: '#/settings'
	}
];
