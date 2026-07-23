<script>
	import { Upload } from '@lucide/svelte';
	import { fade } from 'svelte/transition';
</script>

<div
	transition:fade={{ duration: 200 }}
	class="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-md"
>
	<div
		class="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-accent/40 bg-background/60 p-12 shadow-2xl backdrop-blur-xl transition-all duration-300 shadow-[0_0_30px_var(--glow-color)]/20 dark:shadow-[0_0_40px_var(--glow-color-dark)]/15"
	>
		<Upload class="mb-4 h-12 w-12 text-accent animate-bounce" />

		<p class="text-lg font-display font-semibold text-foreground">Attach files</p>

		<p class="text-sm text-muted-foreground mt-1">Drop your files here to upload</p>
	</div>
</div>
