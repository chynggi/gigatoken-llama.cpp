<script lang="ts">
	import { goto } from '$app/navigation';
	import { Search, Zap, Sliders, Globe, MessageSquare } from '@lucide/svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import { conversations } from '$lib/stores/conversations.svelte';
	import { skillsStore } from '$lib/stores/skills.svelte';
	import { presetsStore } from '$lib/stores/presets.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { RouterService } from '$lib/services/router.service';
	import { ROUTES, SETTINGS_KEYS } from '$lib/constants';
	import { browser } from '$app/environment';
	import { onMount } from 'svelte';
	import { setPendingComposeText } from '$lib/utils/compose-draft';
	import {
		buildCommandPaletteItems,
		filterCommandPaletteItems,
		groupCommandPaletteItems,
		type CommandPaletteIconId,
		type CommandPaletteItemDesc
	} from '$lib/utils/command-palette-commands';
	import type { Component } from 'svelte';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(false), onOpenChange }: Props = $props();

	let searchQuery = $state('');
	let selectedIndex = $state(0);
	let inputRef = $state<HTMLInputElement | null>(null);

	const paletteEnabled = $derived(Boolean(config()[SETTINGS_KEYS.COMMAND_PALETTE_ENABLED]));

	const ICON_MAP: Record<CommandPaletteIconId, Component> = {
		message: MessageSquare,
		zap: Zap,
		sliders: Sliders,
		globe: Globe,
		search: Search
	};

	let allCommands = $derived.by((): CommandPaletteItemDesc[] =>
		buildCommandPaletteItems({
			conversations: conversations().map((c) => ({
				id: c.id,
				name: c.name,
				archived: c.archived
			})),
			skillsEnabled: skillsStore.enabled,
			skills: skillsStore.skills.map((s) => ({
				id: s.id,
				name: s.name,
				description: s.description
			})),
			presetsEnabled: presetsStore.enabled,
			presets: presetsStore.presets.map((p) => ({ id: p.id, name: p.name })),
			webSearchEnabled: Boolean(config()[SETTINGS_KEYS.WEB_SEARCH_ENABLED]),
			routes: {
				start: ROUTES.START,
				newChat: ROUTES.NEW_CHAT,
				settings: ROUTES.SETTINGS,
				settingsPacks: `${ROUTES.SETTINGS}/packs`,
				skills: ROUTES.SKILLS,
				presets: ROUTES.PRESETS,
				searchProviders: ROUTES.SEARCH_PROVIDERS,
				chatHref: (id: string) => RouterService.chat(id)
			}
		})
	);

	let filteredCommands = $derived.by(() => filterCommandPaletteItems(allCommands, searchQuery));

	let groupedCommands = $derived.by(() => groupCommandPaletteItems(filteredCommands));

	$effect(() => {
		if (open) {
			searchQuery = '';
			selectedIndex = 0;
			setTimeout(() => inputRef?.focus(), 50);
		}
	});

	$effect(() => {
		void filteredCommands.length;
		void searchQuery;
		selectedIndex = 0;
	});

	function runAction(cmd: CommandPaletteItemDesc) {
		const action = cmd.action;
		if (action.type === 'goto') {
			goto(action.href);
			return;
		}
		if (action.type === 'compose') {
			setPendingComposeText(action.text);
			if (action.href) goto(action.href);
			return;
		}
		if (action.type === 'apply-preset') {
			presetsStore.applyPreset(action.presetId);
		}
	}

	function handleSelect(cmd: CommandPaletteItemDesc) {
		open = false;
		onOpenChange?.(false);
		runAction(cmd);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'ArrowDown') {
			e.preventDefault();
			selectedIndex = Math.min(selectedIndex + 1, filteredCommands.length - 1);
		} else if (e.key === 'ArrowUp') {
			e.preventDefault();
			selectedIndex = Math.max(selectedIndex - 1, 0);
		} else if (e.key === 'Enter') {
			e.preventDefault();
			const cmd = filteredCommands[selectedIndex];
			if (cmd) handleSelect(cmd);
		} else if (e.key === 'Escape') {
			open = false;
			onOpenChange?.(false);
		}
	}

	function handleGlobalKeydown(e: KeyboardEvent) {
		if (!paletteEnabled) return;
		if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
			e.preventDefault();
			open = !open;
			if (!open) onOpenChange?.(false);
		}
	}

	onMount(() => {
		if (browser) {
			document.addEventListener('keydown', handleGlobalKeydown);
		}
		return () => {
			if (browser) {
				document.removeEventListener('keydown', handleGlobalKeydown);
			}
		};
	});
</script>

{#if paletteEnabled}
	<Dialog.Root bind:open>
		<Dialog.Content class="sm:max-w-lg p-0 gap-0">
			<div class="flex items-center border-b px-3">
				<Search class="mr-2 h-4 w-4 shrink-0 text-muted-foreground" />
				<input
					bind:this={inputRef}
					bind:value={searchQuery}
					class="flex h-11 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground"
					placeholder="Search conversations, skills, presets..."
					onkeydown={handleKeydown}
				/>
			</div>

			<div class="max-h-80 overflow-y-auto p-1">
				{#each Object.entries(groupedCommands) as [category, commands]}
					<div class="px-2 py-1.5 text-xs font-medium text-muted-foreground">
						{category}
					</div>
					{#each commands as cmd}
						{@const globalIdx = filteredCommands.indexOf(cmd)}
						{@const Icon = ICON_MAP[cmd.icon]}
						<button
							class="relative flex w-full cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground {globalIdx ===
							selectedIndex
								? 'bg-accent text-accent-foreground'
								: ''}"
							onclick={() => handleSelect(cmd)}
						>
							<Icon class="h-4 w-4 shrink-0 text-muted-foreground" />
							<span class="flex-1 truncate">{cmd.label}</span>
							{#if cmd.description}
								<span class="text-xs text-muted-foreground truncate max-w-[40%]"
									>{cmd.description}</span
								>
							{/if}
						</button>
					{/each}
				{/each}

				{#if filteredCommands.length === 0}
					<div class="px-2 py-6 text-center text-sm text-muted-foreground">No results found</div>
				{/if}
			</div>

			<div class="flex items-center gap-2 border-t px-3 py-2 text-xs text-muted-foreground">
				<span class="flex items-center gap-1">
					<kbd class="rounded border bg-muted px-1 py-0.5 text-[10px]">Esc</kbd> close
				</span>
				<span class="flex items-center gap-1">
					<kbd class="rounded border bg-muted px-1 py-0.5 text-[10px]">&#8593;&#8595;</kbd> navigate
				</span>
				<span class="flex items-center gap-1">
					<kbd class="rounded border bg-muted px-1 py-0.5 text-[10px]">Enter</kbd> select
				</span>
			</div>
		</Dialog.Content>
	</Dialog.Root>
{/if}
