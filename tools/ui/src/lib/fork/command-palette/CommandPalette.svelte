<script lang="ts">
	import { Globe, MessageSquare, Search, Sliders, Zap } from '@lucide/svelte';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import * as Dialog from '$lib/components/ui/dialog';
	import { ROUTES } from '$lib/constants';
	import {
		buildCommandPaletteItems,
		type CommandPaletteIconId,
		type CommandPaletteItemDesc,
		filterCommandPaletteItems,
		groupCommandPaletteItems
	} from '$lib/fork/command-palette/command-palette-commands';
	import { setPendingComposeText } from '$lib/fork/command-palette/compose-draft';
	import { presetsStore } from '$lib/fork/packs/presets.svelte';
	import { skillsStore } from '$lib/fork/packs/skills.svelte';
	import { FORK_ROUTES } from '$lib/fork/routes';
	import { FORK_SETTINGS_KEYS } from '$lib/fork/settings/keys';
	import { RouterService } from '$lib/services/router.service';
	import { conversationsStore, settingsStore, uiStore } from '$lib/stores';
	import type { Component } from 'svelte';
	import { onMount } from 'svelte';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	let searchQuery = $state('');
	let selectedIndex = $state(0);
	let inputRef = $state<HTMLInputElement | null>(null);

	const paletteEnabled = $derived(
		Boolean(settingsStore.config[FORK_SETTINGS_KEYS.COMMAND_PALETTE_ENABLED])
	);

	const ICON_MAP: Record<CommandPaletteIconId, Component> = {
		globe: Globe,
		message: MessageSquare,
		search: Search,
		sliders: Sliders,
		zap: Zap
	};

	let allCommands = $derived.by((): CommandPaletteItemDesc[] =>
		buildCommandPaletteItems({
			conversations: conversationsStore.conversations.map((c) => ({
				archived: c.archived,
				id: c.id,
				name: c.name
			})),
			presets: presetsStore.presets.map((p) => ({ id: p.id, name: p.name })),
			presetsEnabled: presetsStore.enabled,
			routes: {
				chatHref: (id: string) => RouterService.chat(id),
				newChat: ROUTES.NEW_CHAT,
				presets: FORK_ROUTES.PRESETS,
				searchProviders: FORK_ROUTES.SEARCH_PROVIDERS,
				skills: FORK_ROUTES.SKILLS,
				start: ROUTES.START
			},
			skills: skillsStore.skills.map((s) => ({
				description: s.description,
				id: s.id,
				name: s.name
			})),
			skillsEnabled: skillsStore.enabled,
			webSearchEnabled: Boolean(settingsStore.config[FORK_SETTINGS_KEYS.WEB_SEARCH_ENABLED])
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

		if (action.type === 'open-settings') {
			uiStore.openSettings(action.section);

			return;
		}

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
					onkeydown={handleKeydown}
					placeholder="Search conversations, skills, presets..."
				/>
			</div>

			<div class="max-h-80 overflow-y-auto p-1">
				{#each Object.entries(groupedCommands) as [category, commands] (category)}
					<div class="px-2 py-1.5 text-xs font-medium text-muted-foreground">
						{category}
					</div>

					{#each commands as cmd (cmd.id)}
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
