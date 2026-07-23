<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
	import ChevronDown from '@lucide/svelte/icons/chevron-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { cn } from '$lib/components/ui/utils';
	import { createAutoScrollController } from '$lib/hooks/use-auto-scroll.svelte';
	import { useThrottle } from '$lib/hooks/use-throttle.svelte';
	import { formatReasoningPreview } from '$lib/utils';
	import { config } from '$lib/stores/settings.svelte';
	import { slide } from 'svelte/transition';
	import type { Snippet } from 'svelte';
	import type { Component } from 'svelte';

	interface Props {
		open?: boolean;
		class?: string;
		icon?: Component;
		iconClass?: string;
		iconUrl?: string | null;
		title?: string;
		titleSnippet?: Snippet;
		subtitle?: string;
		shimmerTitle?: boolean;
		preview?: string;
		rawContent?: string;
		isStreaming?: boolean;
		onToggle?: () => void;
		children: Snippet;
	}

	let {
		open = $bindable(false),
		class: className = '',
		icon: IconComponent,
		iconClass = ICON_CLASS_DEFAULT,
		iconUrl = null,
		title = '',
		titleSnippet,
		subtitle,
		shimmerTitle = false,
		preview,
		rawContent,
		isStreaming = false,
		onToggle,
		children
	}: Props = $props();

	let contentContainer: HTMLDivElement | undefined = $state();

	const showThoughtInProgress = $derived(config().showThoughtInProgress as boolean);

	let previewKey = useThrottle(() => rawContent ?? preview ?? '', 500);
	let displayedPreview = $state('');
	let displayedOverflow = $state(0);

	let elapsedSeconds = $state(0);
	let timerInterval: ReturnType<typeof setInterval> | undefined = $state();
	let startTime = $state<number | null>(null);

	$effect(() => {
		if (isStreaming && title.startsWith('Reasoning')) {
			if (!startTime) {
				startTime = Date.now();
				elapsedSeconds = 0;
			}
			if (!timerInterval) {
				timerInterval = setInterval(() => {
					if (startTime) {
						elapsedSeconds = Math.round((Date.now() - startTime) / 100) / 10;
					}
				}, 100);
			}
		} else {
			if (timerInterval) {
				clearInterval(timerInterval);
				timerInterval = undefined;
			}
		}

		return () => {
			if (timerInterval) {
				clearInterval(timerInterval);
			}
		};
	});

	$effect(() => {
		void previewKey.key;
		const content = rawContent ?? preview ?? '';
		const result = formatReasoningPreview(content);
		displayedPreview = result.preview;
		displayedOverflow = result.overflow;
	});

	const autoScroll = createAutoScrollController();

	$effect(() => {
		autoScroll.setContainer(contentContainer);
	});

	$effect(() => {
		autoScroll.updateInterval(open && isStreaming);
	});

	function handleScroll() {
		autoScroll.handleScroll();
	}

	function hideBrokenIcon(event: Event) {
		(event.currentTarget as HTMLImageElement).style.display = 'none';
	}
</script>

<Collapsible.Root
	{open}
	onOpenChange={(value) => {
		open = value;
		onToggle?.();
	}}
	class={cn('group/collapsible', 'my-0!', className)}
>
	<Collapsible.Trigger
		class={cn(
			'flex w-full cursor-pointer items-start justify-between gap-2 text-left',
			'rounded-xl p-2 transition-all duration-300 backdrop-blur-md',
			'bg-muted/10 border border-muted',
			open && 'border-accent/40 bg-muted/20 shadow-[0_0_15px_-3px_var(--glow-color)] dark:shadow-[0_0_20px_-3px_var(--glow-color-dark)]'
		)}
	>
		<div class="flex min-w-0 flex-1 flex-col gap-0.5">
			<div class="flex min-w-0 items-start gap-2 text-muted-foreground">
				{#if iconUrl}
					<img
						src={iconUrl}
						alt=""
						class={cn('shrink-0 rounded-sm mt-0.75', iconClass)}
						onerror={hideBrokenIcon}
					/>
				{:else if IconComponent}
					<IconComponent class={cn('shrink-0 text-muted-foreground/60 mt-0.75', iconClass)} />
				{/if}

				<span class={cn('text-sm font-medium font-mono', shimmerTitle ? 'shimmer-text' : 'text-foreground/80')}>
					{#if titleSnippet}
						{@render titleSnippet()}
					{:else}
						{title}
					{/if}
				</span>

				{#if subtitle}
					<span class="text-xs italic text-muted-foreground/70">{subtitle}</span>
				{:else if title.startsWith('Reasoning') && elapsedSeconds > 0}
					<span class="text-xs italic text-muted-foreground/80">
						{#if isStreaming}
							thinking... ({elapsedSeconds.toFixed(1)}s)
						{:else}
							thought for {elapsedSeconds.toFixed(1)}s
						{/if}
					</span>
				{/if}
			</div>

			{#if displayedPreview && !showThoughtInProgress}
				<div class="flex min-w-0 items-baseline justify-between gap-2 pl-1">
					<div class="w-3/4 truncate text-xs text-muted-foreground/80">
						{displayedPreview}
					</div>
					{#if displayedOverflow > 0}
						<span class="shrink-0 text-xs text-muted-foreground/60">
							{displayedOverflow}+ chars
						</span>
					{/if}
				</div>
			{/if}
		</div>

		<ChevronDown
			class={cn(
				'size-4 shrink-0 text-muted-foreground/60 transition-all duration-150 ease-out opacity-0 group-hover/collapsible:opacity-100 mt-0.75',
				open && 'rotate-180'
			)}
		/>

		<span class="sr-only">Toggle content</span>
	</Collapsible.Trigger>

	<Collapsible.Content forceMount>
		{#if open}
			<div
				bind:this={contentContainer}
				transition:slide={{ duration: 250 }}
				class="overflow-y-auto border-t border-muted px-3 pb-3"
				onscroll={handleScroll}
				style="min-height: var(--min-message-height); max-height: var(--max-message-height);"
			>
				{@render children()}
			</div>
		{/if}
	</Collapsible.Content>
</Collapsible.Root>
