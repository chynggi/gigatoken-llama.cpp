<script lang="ts">
	import { ArrowLeft, Plus, Trash2, Pencil, Save } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { Button } from '$lib/components/ui/button';
	import { skillsStore } from '$lib/stores/skills.svelte';
	import { ROUTES } from '$lib/constants';
	import { onMount } from 'svelte';

	const packEnabled = $derived(skillsStore.enabled);

	let editingId = $state<string | null>(null);
	let editName = $state('');
	let editContent = $state('');
	let editDescription = $state('');
	let editCategory = $state('');
	let showAddForm = $state(false);
	let newName = $state('');
	let newContent = $state('');
	let newDescription = $state('');
	let newCategory = $state('');

	let skills = $derived(skillsStore.skills);

	function startEdit(skill: DatabaseSkill) {
		editingId = skill.id;
		editName = skill.name;
		editContent = skill.content;
		editDescription = skill.description;
		editCategory = skill.category || '';
	}

	async function saveEdit() {
		if (!editingId) return;
		await skillsStore.updateSkill(editingId, {
			name: editName,
			content: editContent,
			description: editDescription,
			category: editCategory || undefined
		});
		editingId = null;
	}

	async function handleDelete(id: string) {
		if (confirm('Delete this skill?')) {
			await skillsStore.deleteSkill(id);
		}
	}

	async function handleAdd() {
		if (!newName.trim() || !newContent.trim()) return;
		await skillsStore.createSkill({
			name: newName.trim(),
			description: newDescription.trim() || newName.trim(),
			content: newContent.trim(),
			category: newCategory.trim() || undefined,
			isBuiltIn: false
		});
		newName = '';
		newContent = '';
		newDescription = '';
		newCategory = '';
		showAddForm = false;
	}
</script>

<div class="p-6 max-w-3xl mx-auto">
	<div class="flex items-center gap-3 mb-6">
		<Button variant="ghost" size="icon" onclick={() => goto(`${ROUTES.SETTINGS}/packs`)}>
			<ArrowLeft class="h-4 w-4" />
		</Button>
		<h1 class="text-xl font-semibold">Skills Library</h1>
	</div>

	{#if !packEnabled}
		<div class="rounded-lg border border-dashed p-6 text-center mb-4">
			<p class="text-sm text-muted-foreground mb-3">
				Skill system is disabled. Enable it under Settings - Packs.
			</p>
			<Button size="sm" onclick={() => goto(`${ROUTES.SETTINGS}/packs`)}>Open Packs settings</Button>
		</div>
	{:else}
	<div class="mb-4">
		<Button variant="outline" size="sm" onclick={() => (showAddForm = !showAddForm)}>
			<Plus class="h-3.5 w-3.5 mr-1" />
			Add Skill
		</Button>
	</div>

	{#if showAddForm}
		<div class="rounded-lg border p-4 mb-4 space-y-3">
			<input type="text" bind:value={newName} placeholder="Skill name (e.g. summarize)" class="w-full rounded-md border px-3 py-2 text-sm" />
			<input type="text" bind:value={newDescription} placeholder="Description" class="w-full rounded-md border px-3 py-2 text-sm" />
			<input type="text" bind:value={newCategory} placeholder="Category (writing, coding, analysis, reasoning)" class="w-full rounded-md border px-3 py-2 text-sm" />
			<textarea bind:value={newContent} placeholder={'Template content. Use {{placeholder}} for variables.'} rows={4} class="w-full rounded-md border px-3 py-2 text-sm font-mono"></textarea>
			<div class="flex gap-2">
				<Button size="sm" onclick={handleAdd}>Save</Button>
				<Button variant="ghost" size="sm" onclick={() => (showAddForm = false)}>Cancel</Button>
			</div>
		</div>
	{/if}

	<div class="space-y-2">
		{#each skills as skill (skill.id)}
			<div class="rounded-lg border p-4">
				{#if editingId === skill.id}
					<div class="space-y-3">
						<input type="text" bind:value={editName} class="w-full rounded-md border px-3 py-2 text-sm" />
						<input type="text" bind:value={editDescription} class="w-full rounded-md border px-3 py-2 text-sm" />
						<input type="text" bind:value={editCategory} class="w-full rounded-md border px-3 py-2 text-sm" />
						<textarea bind:value={editContent} rows={4} class="w-full rounded-md border px-3 py-2 text-sm font-mono"></textarea>
						<div class="flex gap-2">
							<Button size="sm" onclick={saveEdit}><Save class="h-3.5 w-3.5 mr-1" />Save</Button>
							<Button variant="ghost" size="sm" onclick={() => (editingId = null)}>Cancel</Button>
						</div>
					</div>
				{:else}
					<div class="flex items-start justify-between">
						<div class="flex-1">
							<div class="flex items-center gap-2">
								<span class="font-medium text-sm">/{skill.name}</span>
								{#if skill.isBuiltIn}
									<span class="text-[10px] bg-muted rounded-full px-2 py-0.5">built-in</span>
								{/if}
								{#if skill.category}
									<span class="text-[10px] text-muted-foreground">{skill.category}</span>
								{/if}
							</div>
							<p class="text-xs text-muted-foreground mt-1">{skill.description}</p>
							<pre class="text-xs mt-2 bg-muted rounded p-2 overflow-x-auto">{skill.content}</pre>
						</div>
						{#if !skill.isBuiltIn}
							<div class="flex gap-1 ml-2">
								<Button variant="ghost" size="icon" class="h-7 w-7" onclick={() => startEdit(skill)}>
									<Pencil class="h-3.5 w-3.5" />
								</Button>
								<Button variant="ghost" size="icon" class="h-7 w-7 text-destructive" onclick={() => handleDelete(skill.id)}>
									<Trash2 class="h-3.5 w-3.5" />
								</Button>
							</div>
						{/if}
					</div>
				{/if}
			</div>
		{/each}
	</div>
	{/if}
</div>
