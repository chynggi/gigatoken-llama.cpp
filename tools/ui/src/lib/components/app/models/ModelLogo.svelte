<script lang="ts">
	import { Package } from '@lucide/svelte';
	import { parseModelFamily } from '$lib/utils/model-family';
	import { mode } from 'mode-watcher';
	import { ColorMode } from '$lib/enums';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		class?: string;
		model?: string | null;
		/** GGUF general.architecture from /props or model meta (preferred) */
		architecture?: string | null;
		chatTemplate?: string | null;
		modelAlias?: string | null;
		size?: 'sm' | 'md' | 'lg';
		showTooltip?: boolean;
	}

	let {
		class: className = '',
		model = null,
		architecture = null,
		chatTemplate = null,
		modelAlias = null,
		size = 'md',
		showTooltip = true
	}: Props = $props();

	let loadFailed = $state(false);

	const resolved = $derived(
		parseModelFamily(model, { architecture, chatTemplate, modelAlias })
	);
	const isDark = $derived(mode.current === ColorMode.DARK);
	const logoSrc = $derived(isDark ? resolved.logoUrlMono : resolved.logoUrl);
	const sizeClass = $derived(
		size === 'sm' ? 'h-6 w-6' : size === 'lg' ? 'h-10 w-10' : 'h-8 w-8'
	);
	const iconSizeClass = $derived(
		size === 'sm' ? 'h-3 w-3' : size === 'lg' ? 'h-5 w-5' : 'h-4 w-4'
	);

	$effect(() => {
		// Reset failure when the source model / arch changes
		void logoSrc;
		loadFailed = false;
	});
</script>

{#snippet avatar()}
	<div
		class="model-response-logo relative flex shrink-0 items-center justify-center overflow-hidden rounded-full border border-border/60 bg-muted/40 shadow-sm backdrop-blur-sm {sizeClass} {className}"
		aria-hidden={!showTooltip}
	>
		{#if !loadFailed && logoSrc}
			<img
				src={logoSrc}
				alt=""
				class="h-[70%] w-[70%] object-contain"
				loading="lazy"
				referrerpolicy="no-referrer"
				onerror={() => {
					loadFailed = true;
				}}
			/>
		{:else}
			<Package class="text-muted-foreground {iconSizeClass}" />
		{/if}
	</div>
{/snippet}

{#if showTooltip}
	<Tooltip.Root>
		<Tooltip.Trigger>
			{#snippet child({ props })}
				<div {...props} class="inline-flex">
					{@render avatar()}
				</div>
			{/snippet}
		</Tooltip.Trigger>
		<Tooltip.Content>
			{resolved.definition.label}{architecture
				? ` (${architecture})`
				: ''}{model ? ` · ${model}` : ''}
		</Tooltip.Content>
	</Tooltip.Root>
{:else}
	{@render avatar()}
{/if}
