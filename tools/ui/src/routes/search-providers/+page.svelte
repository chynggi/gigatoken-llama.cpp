<script lang="ts">
	import { ArrowLeft, Globe, Plus, ToggleLeft, ToggleRight, Trash2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import type { SearchProviderType } from '$lib/fork/db/types';
	import { searchProvidersStore } from '$lib/fork/packs/search-providers.svelte';
	import { SEARCH_PROVIDERS } from '$lib/fork/search/search';
	import { uiStore } from '$lib/stores';

	const packEnabled = $derived(searchProvidersStore.enabled);

	let showAddForm = $state(false);
	let selectedType = $state<SearchProviderType>('searxng');
	let newApiKey = $state('');
	let newBaseUrl = $state('');

	let providers = $derived(searchProvidersStore.providers);
	let availableTypes = Object.entries(SEARCH_PROVIDERS).map(([id, p]) => ({
		id,
		name: p.name,
		requiresApiKey: p.requiresApiKey,
		requiresBaseUrl: p.requiresBaseUrl
	}));

	async function handleAdd() {
		const info = SEARCH_PROVIDERS[selectedType];

		if (!info) return;

		if (info.requiresApiKey && !newApiKey.trim()) return;

		if (info.requiresBaseUrl && !newBaseUrl.trim()) return;

		await searchProvidersStore.addProvider({
			apiKey: newApiKey.trim() || undefined,
			baseUrl: newBaseUrl.trim() || undefined,
			enabled: true,
			name: info.name,
			priority: providers.length,
			type: selectedType
		});

		newApiKey = '';
		newBaseUrl = '';
		showAddForm = false;
	}

	async function handleToggle(id: string) {
		await searchProvidersStore.toggleProvider(id);
	}

	async function handleDelete(id: string) {
		if (confirm('Remove this search provider?')) {
			await searchProvidersStore.deleteProvider(id);
		}
	}
</script>

<div class="p-6 max-w-3xl mx-auto">
	<div class="flex items-center gap-3 mb-6">
		<Button onclick={() => uiStore.openSettings('packs')} size="icon" variant="ghost">
			<ArrowLeft class="h-4 w-4" />
		</Button>

		<h1 class="text-xl font-semibold">Web Search Providers</h1>
	</div>

	{#if !packEnabled}
		<div class="rounded-lg border border-dashed p-6 text-center mb-4">
			<p class="text-sm text-muted-foreground mb-3">
				Web search is disabled. Enable it under Settings - Packs, then add a provider here.
			</p>

			<Button onclick={() => uiStore.openSettings('packs')} size="sm">Open Packs settings</Button>
		</div>
	{:else}
		<div class="mb-4">
			<Button onclick={() => (showAddForm = !showAddForm)} size="sm" variant="outline">
				<Plus class="h-3.5 w-3.5 mr-1" />
				Add Provider
			</Button>
		</div>

		{#if showAddForm}
			<div class="rounded-lg border p-4 mb-4 space-y-3">
				<select
					bind:value={selectedType}
					class="w-full rounded-md border px-3 py-2 text-sm bg-background"
				>
					{#each availableTypes as t (t.id)}
						<option value={t.id}>{t.name}</option>
					{/each}
				</select>

				{#if SEARCH_PROVIDERS[selectedType]?.requiresApiKey}
					<input
						bind:value={newApiKey}
						class="w-full rounded-md border px-3 py-2 text-sm"
						placeholder="API Key"
						type="password"
					/>
				{/if}

				{#if SEARCH_PROVIDERS[selectedType]?.requiresBaseUrl}
					<input
						bind:value={newBaseUrl}
						class="w-full rounded-md border px-3 py-2 text-sm"
						placeholder="Base URL (e.g. http://localhost:8080)"
						type="text"
					/>
				{/if}

				<div class="flex gap-2">
					<Button onclick={handleAdd} size="sm">Add</Button>

					<Button onclick={() => (showAddForm = false)} size="sm" variant="ghost">Cancel</Button>
				</div>
			</div>
		{/if}

		<div class="space-y-2">
			{#each providers as provider (provider.id)}
				<div class="rounded-lg border p-4">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<Globe class="h-4 w-4 text-muted-foreground" />

							<span class="font-medium text-sm">{provider.name}</span>

							<span class="text-[10px] text-muted-foreground bg-muted rounded-full px-2 py-0.5"
								>{provider.type}</span
							>

							{#if provider.enabled}
								<span class="text-[10px] text-green-600">enabled</span>
							{:else}
								<span class="text-[10px] text-muted-foreground">disabled</span>
							{/if}
						</div>

						<div class="flex gap-1">
							<Button
								class="h-7 w-7"
								onclick={() => handleToggle(provider.id)}
								size="icon"
								variant="ghost"
							>
								{#if provider.enabled}
									<ToggleRight class="h-4 w-4 text-green-600" />
								{:else}
									<ToggleLeft class="h-4 w-4 text-muted-foreground" />
								{/if}
							</Button>

							<Button
								class="h-7 w-7 text-destructive"
								onclick={() => handleDelete(provider.id)}
								size="icon"
								variant="ghost"
							>
								<Trash2 class="h-3.5 w-3.5" />
							</Button>
						</div>
					</div>
				</div>
			{/each}

			{#if providers.length === 0}
				<div class="text-center py-8">
					<p class="text-sm text-muted-foreground mb-2">No search providers configured</p>

					<p class="text-xs text-muted-foreground">
						Add SearXNG (self-hosted, free) or Tavily/Brave/Serper (API key required)
					</p>
				</div>
			{/if}
		</div>
	{/if}
</div>
