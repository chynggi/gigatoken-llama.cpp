<script lang="ts">
	import { ArrowLeft, Pencil, Play, Plus, Trash2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { SETTINGS_KEYS } from '$lib/constants';
	import type { DatabasePreset } from '$lib/fork/db/types';
	import {
		collectSamplingParamsFromForm,
		deriveMcpOverridesFromServers,
		parseMcpOverridesJson,
		PRESET_SAMPLING_KEYS,
		type PresetSamplingKey
	} from '$lib/fork/packs/preset-apply';
	import { presetsStore } from '$lib/fork/packs/presets.svelte';
	import { FORK_SETTINGS_KEYS } from '$lib/fork/settings/keys';
	import { uiStore } from '$lib/stores';
	import { settingsStore } from '$lib/stores';
	import { toast } from 'svelte-sonner';

	const packEnabled = $derived(presetsStore.enabled);
	const presets = $derived(presetsStore.presets);

	let showAddForm = $state(false);
	let editingId = $state<string | null>(null);

	let name = $state('');
	let systemMessage = $state('');
	let mcpOverridesJson = $state('');
	let webSearchEnabled = $state(false);
	let webSearchProvider = $state('');
	let samplingFields = $state<Partial<Record<PresetSamplingKey, string>>>({});

	const samplingLabels: Record<PresetSamplingKey, string> = {
		[SETTINGS_KEYS.MAX_TOKENS]: 'Max tokens',
		[SETTINGS_KEYS.MIN_P]: 'Min P',
		[SETTINGS_KEYS.REPEAT_PENALTY]: 'Repeat penalty',
		[SETTINGS_KEYS.TEMPERATURE]: 'Temperature',
		[SETTINGS_KEYS.TOP_K]: 'Top K',
		[SETTINGS_KEYS.TOP_P]: 'Top P'
	};

	function resetForm() {
		name = '';
		systemMessage = '';
		mcpOverridesJson = '';
		webSearchEnabled = false;
		webSearchProvider = '';
		samplingFields = {};
		editingId = null;
		showAddForm = false;
	}

	function fillFromCurrentSettings() {
		const c = settingsStore.config;

		systemMessage = String(c[SETTINGS_KEYS.SYSTEM_MESSAGE] ?? '');
		const next: Partial<Record<PresetSamplingKey, string>> = {};

		for (const key of PRESET_SAMPLING_KEYS) {
			const v = c[key];

			if (v !== undefined && v !== null && v !== '') {
				next[key] = String(v);
			}
		}
		samplingFields = next;
		const rawServers = c[SETTINGS_KEYS.MCP_SERVERS];
		const currentOverrides =
			typeof rawServers === 'string' ? deriveMcpOverridesFromServers(rawServers) : undefined;

		mcpOverridesJson = currentOverrides ? JSON.stringify(currentOverrides, null, 2) : '';
		webSearchEnabled = Boolean(c[FORK_SETTINGS_KEYS.WEB_SEARCH_ENABLED]);
		webSearchProvider = String(c[FORK_SETTINGS_KEYS.WEB_SEARCH_ACTIVE_PROVIDER] ?? '');
	}

	function startCreate() {
		resetForm();
		showAddForm = true;
		fillFromCurrentSettings();
		name = '';
	}

	function startEdit(preset: DatabasePreset) {
		editingId = preset.id;
		showAddForm = true;
		name = preset.name;
		systemMessage = preset.systemMessage ?? '';
		mcpOverridesJson = preset.mcpOverrides ? JSON.stringify(preset.mcpOverrides, null, 2) : '';
		webSearchEnabled = Boolean(preset.webSearchEnabled);
		webSearchProvider = preset.webSearchProvider ?? '';
		const next: Partial<Record<PresetSamplingKey, string>> = {};

		if (preset.samplingParams) {
			for (const key of PRESET_SAMPLING_KEYS) {
				const v = preset.samplingParams[key];

				if (v !== undefined) next[key] = String(v);
			}
		}

		samplingFields = next;
	}

	function buildPayload(): Omit<DatabasePreset, 'id' | 'createdAt'> | null {
		if (!name.trim()) {
			toast.error('Name is required');

			return null;
		}

		const mcpParsed = parseMcpOverridesJson(mcpOverridesJson);

		if (mcpParsed === null) {
			toast.error('MCP overrides must be a JSON array of { serverId, enabled }');

			return null;
		}

		return {
			mcpOverrides: mcpParsed,
			name: name.trim(),
			samplingParams: collectSamplingParamsFromForm(samplingFields),
			systemMessage: systemMessage.trim() || undefined,
			webSearchEnabled,
			webSearchProvider: webSearchProvider.trim() || undefined
		};
	}

	async function handleSave() {
		const payload = buildPayload();

		if (!payload) return;

		if (editingId) {
			await presetsStore.updatePreset(editingId, payload);
			toast.success('Preset updated');
		} else {
			await presetsStore.createPreset(payload);
			toast.success('Preset created');
		}

		resetForm();
	}

	async function handleDelete(id: string) {
		if (confirm('Delete this preset?')) {
			await presetsStore.deletePreset(id);
		}
	}
</script>

<div class="p-6 max-w-3xl mx-auto">
	<div class="flex items-center gap-3 mb-6">
		<Button onclick={() => uiStore.openSettings('packs')} size="icon" variant="ghost">
			<ArrowLeft class="h-4 w-4" />
		</Button>

		<h1 class="text-xl font-semibold">Chat Presets</h1>
	</div>

	{#if !packEnabled}
		<div class="rounded-lg border border-dashed p-6 text-center mb-4">
			<p class="text-sm text-muted-foreground mb-3">
				Chat presets are disabled. Enable them under Settings - Packs.
			</p>

			<Button onclick={() => uiStore.openSettings('packs')} size="sm">Open Packs settings</Button>
		</div>
	{:else}
		<div class="mb-4">
			<Button onclick={startCreate} size="sm" variant="outline">
				<Plus class="h-3.5 w-3.5 mr-1" />
				New Preset
			</Button>
		</div>

		{#if showAddForm}
			<div class="rounded-lg border p-4 mb-4 space-y-3">
				<input
					bind:value={name}
					class="w-full rounded-md border px-3 py-2 text-sm"
					placeholder="Preset name"
					type="text"
				/>

				<textarea
					bind:value={systemMessage}
					class="w-full rounded-md border px-3 py-2 text-sm"
					placeholder="System message (optional)"
					rows={3}
				></textarea>

				<div class="grid grid-cols-2 gap-2">
					{#each PRESET_SAMPLING_KEYS as key (key)}
						<label class="flex flex-col gap-1 text-xs text-muted-foreground">
							{samplingLabels[key]}
							<input
								class="rounded-md border px-2 py-1.5 text-sm text-foreground"
								oninput={(e) => {
									samplingFields = {
										...samplingFields,
										[key]: (e.currentTarget as HTMLInputElement).value
									};
								}}
								placeholder="(optional)"
								step="any"
								type="number"
								value={samplingFields[key] ?? ''}
							/>
						</label>
					{/each}
				</div>

				<label class="flex flex-col gap-1 text-xs text-muted-foreground">
					MCP default overrides (JSON array)
					<textarea
						bind:value={mcpOverridesJson}
						class="w-full rounded-md border px-3 py-2 text-sm font-mono text-foreground"
						placeholder={'[{"serverId":"...","enabled":true}]'}
						rows={3}
					></textarea>
				</label>

				<label class="flex items-center gap-2 text-sm">
					<input bind:checked={webSearchEnabled} type="checkbox" />
					Web search enabled
				</label>

				<input
					bind:value={webSearchProvider}
					class="w-full rounded-md border px-3 py-2 text-sm"
					placeholder="Active search provider id (optional)"
					type="text"
				/>

				<div class="flex gap-2">
					<Button onclick={handleSave} size="sm">{editingId ? 'Update' : 'Save'}</Button>

					<Button onclick={resetForm} size="sm" variant="ghost">Cancel</Button>
				</div>
			</div>
		{/if}

		<div class="space-y-2">
			{#each presets as preset (preset.id)}
				<div class="rounded-lg border p-4">
					<div class="flex items-start justify-between">
						<div class="flex-1 min-w-0">
							<span class="font-medium text-sm">{preset.name}</span>

							{#if preset.systemMessage}
								<p class="text-xs text-muted-foreground mt-1 line-clamp-2">
									{preset.systemMessage}
								</p>
							{/if}

							<div class="mt-1 flex flex-wrap gap-1 text-[10px] text-muted-foreground">
								{#if preset.samplingParams}
									<span class="rounded bg-muted px-1.5 py-0.5">sampling</span>
								{/if}

								{#if preset.mcpOverrides}
									<span class="rounded bg-muted px-1.5 py-0.5">mcp</span>
								{/if}

								{#if preset.webSearchEnabled}
									<span class="rounded bg-muted px-1.5 py-0.5">web search</span>
								{/if}
							</div>
						</div>

						<div class="flex gap-1 ml-2">
							<Button
								class="h-7 w-7"
								onclick={() => presetsStore.applyPreset(preset.id)}
								size="icon"
								title="Apply preset"
								variant="ghost"
							>
								<Play class="h-3.5 w-3.5" />
							</Button>

							<Button
								class="h-7 w-7"
								onclick={() => startEdit(preset)}
								size="icon"
								title="Edit"
								variant="ghost"
							>
								<Pencil class="h-3.5 w-3.5" />
							</Button>

							<Button
								class="h-7 w-7 text-destructive"
								onclick={() => handleDelete(preset.id)}
								size="icon"
								variant="ghost"
							>
								<Trash2 class="h-3.5 w-3.5" />
							</Button>
						</div>
					</div>
				</div>
			{/each}

			{#if presets.length === 0}
				<p class="text-sm text-muted-foreground text-center py-8">
					No presets yet. Save system message, sampling, MCP, and web search as a preset.
				</p>
			{/if}
		</div>
	{/if}
</div>
