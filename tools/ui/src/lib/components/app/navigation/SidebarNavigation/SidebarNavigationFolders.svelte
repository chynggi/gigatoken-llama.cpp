<script lang="ts">
	import { Folder, FolderPlus, MoreHorizontal, Pencil, Trash2 } from '@lucide/svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { DropdownMenuActions } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { fade } from 'svelte/transition';

	interface Props {
		activeFolderId?: string;
		onFolderSelect?: (folderId: string | undefined) => void;
	}

	let { activeFolderId, onFolderSelect }: Props = $props();

	let isCreating = $state(false);
	let newFolderName = $state('');
	let editingFolderId = $state<string | null>(null);
	let editName = $state('');

	let folders = $derived(conversationsStore.folders);

	function handleCreate() {
		const name = newFolderName.trim();
		if (!name) {
			isCreating = false;
			newFolderName = '';
			return;
		}
		conversationsStore.createFolder(name);
		isCreating = false;
		newFolderName = '';
	}

	function startEdit(folder: DatabaseFolder) {
		editingFolderId = folder.id;
		editName = folder.name;
	}

	function commitEdit() {
		const name = editName.trim();
		if (name && editingFolderId) {
			conversationsStore.renameFolder(editingFolderId, name);
		}
		editingFolderId = null;
		editName = '';
	}

	function handleDelete(folderId: string) {
		conversationsStore.deleteFolder(folderId);
		if (activeFolderId === folderId) {
			onFolderSelect?.(undefined);
		}
	}
</script>

<div class="flex flex-col gap-1">
	<div class="flex items-center justify-between px-2 py-1">
		<span class="text-xs font-medium text-muted-foreground">Folders</span>
		<Button
			variant="ghost"
			size="icon"
			class="h-6 w-6"
			onclick={() => {
				isCreating = !isCreating;
				newFolderName = '';
			}}
		>
			<FolderPlus class="h-3.5 w-3.5" />
		</Button>
	</div>

	{#if isCreating}
		<div class="px-2" transition:fade={{ duration: 150 }}>
			<input
				type="text"
				bind:value={newFolderName}
				class="w-full rounded-md border bg-background px-2 py-1 text-xs outline-none focus:border-ring"
				placeholder="Folder name..."
				onkeydown={(e) => {
					if (e.key === 'Enter') handleCreate();
					if (e.key === 'Escape') {
						isCreating = false;
						newFolderName = '';
					}
				}}
				onblur={handleCreate}
			/>
		</div>
	{/if}

	<ul class="flex flex-col gap-0.5">
		<li>
			<Button
				variant="ghost"
				size="sm"
				class="w-full justify-start gap-2 px-2 h-7 text-xs {!activeFolderId
					? 'bg-accent text-accent-foreground'
					: ''}"
				onclick={() => onFolderSelect?.(undefined)}
			>
				<Folder class="h-3.5 w-3.5" />
				<span class="truncate">All Conversations</span>
			</Button>
		</li>

		{#each folders as folder (folder.id)}
			<li class="group/folder flex items-center gap-0.5">
				{#if editingFolderId === folder.id}
					<div class="flex-1 px-2" transition:fade={{ duration: 100 }}>
						<input
							type="text"
							bind:value={editName}
							class="w-full rounded-md border bg-background px-2 py-1 text-xs outline-none focus:border-ring"
							onkeydown={(e) => {
								if (e.key === 'Enter') commitEdit();
								if (e.key === 'Escape') {
									editingFolderId = null;
								}
							}}
							onblur={commitEdit}
						/>
					</div>
				{:else}
					<Button
						variant="ghost"
						size="sm"
						class="min-w-0 flex-1 justify-start gap-2 px-2 h-7 text-xs {activeFolderId ===
						folder.id
							? 'bg-accent text-accent-foreground'
							: ''}"
						onclick={() => onFolderSelect?.(folder.id)}
					>
						<span
							class="h-2.5 w-2.5 rounded-full shrink-0"
							style="background-color: {folder.color || 'var(--muted-foreground)'}"
						></span>
						<span class="truncate">{folder.name}</span>
					</Button>
					<div class="shrink-0 opacity-0 group-hover/folder:opacity-100 focus-within:opacity-100">
						<DropdownMenuActions
							triggerIcon={MoreHorizontal}
							triggerTooltip="Folder actions"
							actions={[
								{
									icon: Pencil,
									label: 'Rename',
									onclick: (e: Event) => {
										e.stopPropagation();
										startEdit(folder);
									}
								},
								{
									icon: Trash2,
									label: 'Delete',
									variant: 'destructive',
									separator: true,
									onclick: (e: Event) => {
										e.stopPropagation();
										handleDelete(folder.id);
									}
								}
							]}
						/>
					</div>
				{/if}
			</li>
		{/each}
	</ul>
</div>
