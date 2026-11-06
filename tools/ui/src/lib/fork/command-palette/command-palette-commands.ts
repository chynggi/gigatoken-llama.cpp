/**
 * Pure builders for command palette items.
 * The shell component only wires navigation/store side effects.
 */

export type CommandPaletteIconId = 'message' | 'zap' | 'sliders' | 'globe' | 'search';

export interface CommandPaletteItemDesc {
	id: string;
	label: string;
	description?: string;
	icon: CommandPaletteIconId;
	category: string;
	/** Discriminated action for the shell to execute */
	action:
		| { type: 'goto'; href: string }
		| { type: 'open-settings'; section?: string }
		| { type: 'compose'; text: string; href?: string }
		| { type: 'apply-preset'; presetId: string };
}

export interface CommandPaletteSources {
	conversations: Array<{ id: string; name: string; archived?: boolean }>;
	skillsEnabled: boolean;
	skills: Array<{ id: string; name: string; description: string }>;
	presetsEnabled: boolean;
	presets: Array<{ id: string; name: string }>;
	webSearchEnabled: boolean;
	routes: {
		start: string;
		newChat: string;
		skills: string;
		presets: string;
		searchProviders: string;
		chatHref: (id: string) => string;
	};
}

/** Build the full unfiltered command list from live store snapshots. */
export function buildCommandPaletteItems(sources: CommandPaletteSources): CommandPaletteItemDesc[] {
	const cmds: CommandPaletteItemDesc[] = [];
	const { routes } = sources;

	for (const conv of sources.conversations) {
		if (conv.archived) continue;

		cmds.push({
			action: { href: routes.chatHref(conv.id), type: 'goto' },
			category: 'Conversations',
			description: 'Open conversation',
			icon: 'message',
			id: `conv-${conv.id}`,
			label: conv.name
		});
	}

	if (sources.skillsEnabled) {
		for (const skill of sources.skills) {
			cmds.push({
				action: {
					href: routes.start,
					text: `/${skill.name} `,
					type: 'compose'
				},
				category: 'Skills',
				description: skill.description,
				icon: 'zap',
				id: `skill-${skill.id}`,
				label: `/${skill.name}`
			});
		}
	}

	if (sources.presetsEnabled) {
		for (const preset of sources.presets) {
			cmds.push({
				action: { presetId: preset.id, type: 'apply-preset' },
				category: 'Presets',
				description: 'Apply chat preset',
				icon: 'sliders',
				id: `preset-${preset.id}`,
				label: preset.name
			});
		}
	}

	cmds.push({
		action: { type: 'open-settings' },
		category: 'Navigation',
		description: 'Open settings',
		icon: 'sliders',
		id: 'settings-general',
		label: 'Settings'
	});

	cmds.push({
		action: { section: 'packs', type: 'open-settings' },
		category: 'Navigation',
		description: 'Enable optional features',
		icon: 'globe',
		id: 'settings-packs',
		label: 'Packs settings'
	});

	if (sources.skillsEnabled) {
		cmds.push({
			action: { href: routes.skills, type: 'goto' },
			category: 'Navigation',
			description: 'Edit skill templates',
			icon: 'zap',
			id: 'manage-skills',
			label: 'Manage skills'
		});
	}

	if (sources.presetsEnabled) {
		cmds.push({
			action: { href: routes.presets, type: 'goto' },
			category: 'Navigation',
			description: 'Edit chat presets',
			icon: 'sliders',
			id: 'manage-presets',
			label: 'Manage presets'
		});
	}

	if (sources.webSearchEnabled) {
		cmds.push({
			action: { href: routes.searchProviders, type: 'goto' },
			category: 'Navigation',
			description: 'Configure search providers',
			icon: 'globe',
			id: 'manage-search-providers',
			label: 'Web search providers'
		});
	}

	cmds.push({
		action: { href: routes.newChat, type: 'goto' },
		category: 'Navigation',
		description: 'Start a fresh chat',
		icon: 'message',
		id: 'new-chat',
		label: 'New conversation'
	});

	return cmds;
}

/** Filter commands by free-text query (label, description, category). */
export function filterCommandPaletteItems(
	items: CommandPaletteItemDesc[],
	query: string
): CommandPaletteItemDesc[] {
	const q = query.toLowerCase().trim();

	if (!q) return items;

	return items.filter(
		(c) =>
			c.label.toLowerCase().includes(q) ||
			(c.description && c.description.toLowerCase().includes(q)) ||
			c.category.toLowerCase().includes(q)
	);
}

/** Group filtered commands by category, preserving order. */
export function groupCommandPaletteItems(
	items: CommandPaletteItemDesc[]
): Record<string, CommandPaletteItemDesc[]> {
	const groups: Record<string, CommandPaletteItemDesc[]> = {};

	for (const cmd of items) {
		if (!groups[cmd.category]) groups[cmd.category] = [];

		groups[cmd.category].push(cmd);
	}

	return groups;
}
