<script lang="ts">
	import { X } from '@lucide/svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';

	let allTags = $derived(conversationsStore.allTags);
	let activeTag = $derived(conversationsStore.activeTagFilter);
</script>

{#if allTags.length > 0}
	<div class="flex flex-col gap-1">
		<div class="flex items-center px-2 py-1">
			<span class="text-xs font-medium text-muted-foreground">Tags</span>
		</div>
		<div class="flex flex-wrap gap-1 px-2">
			{#each allTags as tag}
				<button
					class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium transition-colors {activeTag === tag
						? 'bg-primary text-primary-foreground'
						: 'bg-muted text-muted-foreground hover:bg-muted-foreground/20'}"
					onclick={() => conversationsStore.setTagFilter(tag)}
				>
					{tag}
					{#if activeTag === tag}
						<X class="h-2.5 w-2.5" />
					{/if}
				</button>
			{/each}
		</div>
	</div>
{/if}
