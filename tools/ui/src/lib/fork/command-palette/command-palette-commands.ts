/**
 * Pure builders for command palette items.
 * The shell component only wires navigation/store side effects.
 */

export type CommandPaletteIconId =
	| 'message'
	| 'zap'
	| 'sliders'
	| 'globe'
	| 'search';

export interface CommandPaletteItemDesc {
	id: string;
	label: string;
	description?: string;
	icon: CommandPaletteIconId;
	category: string;
	/** Discriminated action for the shell to execute */
	action:
		| { type: 'goto'; href: string }
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
		settings: string;
		settingsPacks: string;
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
			id: `conv-${conv.id}`,
			label: conv.name,
			description: 'Open conversation',
			icon: 'message',
			category: 'Conversations',
			action: { type: 'goto', href: routes.chatHref(conv.id) }
		});
	}

	if (sources.skillsEnabled) {
		for (const skill of sources.skills) {
			cmds.push({
				id: `skill-${skill.id}`,
				label: `/${skill.name}`,
				description: skill.description,
				icon: 'zap',
				category: 'Skills',
				action: {
					type: 'compose',
					text: `/${skill.name} `,
					href: routes.start
				}
			});
		}
	}

	if (sources.presetsEnabled) {
		for (const preset of sources.presets) {
			cmds.push({
				id: `preset-${preset.id}`,
				label: preset.name,
				description: 'Apply chat preset',
				icon: 'sliders',
				category: 'Presets',
				action: { type: 'apply-preset', presetId: preset.id }
			});
		}
	}

	cmds.push({
		id: 'settings-general',
		label: 'Settings',
		description: 'Open settings',
		icon: 'sliders',
		category: 'Navigation',
		action: { type: 'goto', href: routes.settings }
	});

	cmds.push({
		id: 'settings-packs',
		label: 'Packs settings',
		description: 'Enable optional features',
		icon: 'globe',
		category: 'Navigation',
		action: { type: 'goto', href: routes.settingsPacks }
	});

	if (sources.skillsEnabled) {
		cmds.push({
			id: 'manage-skills',
			label: 'Manage skills',
			description: 'Edit skill templates',
			icon: 'zap',
			category: 'Navigation',
			action: { type: 'goto', href: routes.skills }
		});
	}

	if (sources.presetsEnabled) {
		cmds.push({
			id: 'manage-presets',
			label: 'Manage presets',
			description: 'Edit chat presets',
			icon: 'sliders',
			category: 'Navigation',
			action: { type: 'goto', href: routes.presets }
		});
	}

	if (sources.webSearchEnabled) {
		cmds.push({
			id: 'manage-search-providers',
			label: 'Web search providers',
			description: 'Configure search providers',
			icon: 'globe',
			category: 'Navigation',
			action: { type: 'goto', href: routes.searchProviders }
		});
	}

	cmds.push({
		id: 'new-chat',
		label: 'New conversation',
		description: 'Start a fresh chat',
		icon: 'message',
		category: 'Navigation',
		action: { type: 'goto', href: routes.newChat }
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
