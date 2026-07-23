<script lang="ts">
	import { Globe, Loader2, X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { searchProvidersStore } from '$lib/stores/search-providers.svelte';

	let results = $derived(searchProvidersStore.searchResults);
	let isSearching = $derived(searchProvidersStore.isSearching);
	let lastQuery = $derived(searchProvidersStore.lastQuery);

	function clear() {
		searchProvidersStore.clearResults();
	}
</script>

{#if results.length > 0 || isSearching}
	<div class="rounded-lg border bg-background p-3">
		<div class="flex items-center justify-between mb-2">
			<span class="text-xs font-medium text-muted-foreground flex items-center gap-1">
				<Globe class="h-3 w-3" />
				Web search results
				{#if lastQuery}
					- "{lastQuery}"
				{/if}
			</span>
			<Button variant="ghost" size="icon" class="h-5 w-5" onclick={clear}>
				<X class="h-3 w-3" />
			</Button>
		</div>

		{#if isSearching}
			<div class="flex items-center gap-2 text-xs text-muted-foreground">
				<Loader2 class="h-3 w-3 animate-spin" />
				Searching...
			</div>
		{:else}
			<div class="flex flex-col gap-1 max-h-40 overflow-y-auto">
				{#each results as result}
					<a
						href={result.url}
						target="_blank"
						rel="noopener noreferrer"
						class="block rounded px-2 py-1 text-xs hover:bg-accent transition-colors"
					>
						<div class="font-medium text-foreground truncate">{result.title}</div>
						<div class="text-muted-foreground truncate">{result.snippet}</div>
					</a>
				{/each}
			</div>
		{/if}
	</div>
{/if}
