<script lang="ts">
	import { ArrowLeft, Plus, Trash2, Play, Pencil } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { Button } from '$lib/components/ui/button';
	import { presetsStore } from '$lib/stores/presets.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { ROUTES, SETTINGS_KEYS } from '$lib/constants';
	import {
		PRESET_SAMPLING_KEYS,
		collectSamplingParamsFromForm,
		parseMcpOverridesJson,
		type PresetSamplingKey
	} from '$lib/utils/preset-apply';
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
		[SETTINGS_KEYS.TEMPERATURE]: 'Temperature',
		[SETTINGS_KEYS.TOP_P]: 'Top P',
		[SETTINGS_KEYS.TOP_K]: 'Top K',
		[SETTINGS_KEYS.MIN_P]: 'Min P',
		[SETTINGS_KEYS.MAX_TOKENS]: 'Max tokens',
		[SETTINGS_KEYS.REPEAT_PENALTY]: 'Repeat penalty'
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
		const c = config();
		systemMessage = String(c[SETTINGS_KEYS.SYSTEM_MESSAGE] ?? '');
		const next: Partial<Record<PresetSamplingKey, string>> = {};
		for (const key of PRESET_SAMPLING_KEYS) {
			const v = c[key];
			if (v !== undefined && v !== null && v !== '') {
				next[key] = String(v);
			}
		}
		samplingFields = next;
		mcpOverridesJson = String(c[SETTINGS_KEYS.MCP_DEFAULT_SERVER_OVERRIDES] ?? '');
		webSearchEnabled = Boolean(c[SETTINGS_KEYS.WEB_SEARCH_ENABLED]);
		webSearchProvider = String(c[SETTINGS_KEYS.WEB_SEARCH_ACTIVE_PROVIDER] ?? '');
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
			name: name.trim(),
			systemMessage: systemMessage.trim() || undefined,
			samplingParams: collectSamplingParamsFromForm(samplingFields),
			mcpOverrides: mcpParsed,
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
		<Button variant="ghost" size="icon" onclick={() => goto(`${ROUTES.SETTINGS}/packs`)}>
			<ArrowLeft class="h-4 w-4" />
		</Button>
		<h1 class="text-xl font-semibold">Chat Presets</h1>
	</div>

	{#if !packEnabled}
		<div class="rounded-lg border border-dashed p-6 text-center mb-4">
			<p class="text-sm text-muted-foreground mb-3">
				Chat presets are disabled. Enable them under Settings - Packs.
			</p>
			<Button size="sm" onclick={() => goto(`${ROUTES.SETTINGS}/packs`)}>Open Packs settings</Button>
		</div>
	{:else}
		<div class="mb-4">
			<Button variant="outline" size="sm" onclick={startCreate}>
				<Plus class="h-3.5 w-3.5 mr-1" />
				New Preset
			</Button>
		</div>

		{#if showAddForm}
			<div class="rounded-lg border p-4 mb-4 space-y-3">
				<input
					type="text"
					bind:value={name}
					placeholder="Preset name"
					class="w-full rounded-md border px-3 py-2 text-sm"
				/>
				<textarea
					bind:value={systemMessage}
					placeholder="System message (optional)"
					rows={3}
					class="w-full rounded-md border px-3 py-2 text-sm"
				></textarea>

				<div class="grid grid-cols-2 gap-2">
					{#each PRESET_SAMPLING_KEYS as key}
						<label class="flex flex-col gap-1 text-xs text-muted-foreground">
							{samplingLabels[key]}
							<input
								type="number"
								step="any"
								class="rounded-md border px-2 py-1.5 text-sm text-foreground"
								value={samplingFields[key] ?? ''}
								oninput={(e) => {
									samplingFields = {
										...samplingFields,
										[key]: (e.currentTarget as HTMLInputElement).value
									};
								}}
								placeholder="(optional)"
							/>
						</label>
					{/each}
				</div>

				<label class="flex flex-col gap-1 text-xs text-muted-foreground">
					MCP default overrides (JSON array)
					<textarea
						bind:value={mcpOverridesJson}
						placeholder={'[{"serverId":"...","enabled":true}]'}
						rows={3}
						class="w-full rounded-md border px-3 py-2 text-sm font-mono text-foreground"
					></textarea>
				</label>

				<label class="flex items-center gap-2 text-sm">
					<input type="checkbox" bind:checked={webSearchEnabled} />
					Web search enabled
				</label>

				<input
					type="text"
					bind:value={webSearchProvider}
					placeholder="Active search provider id (optional)"
					class="w-full rounded-md border px-3 py-2 text-sm"
				/>

				<div class="flex gap-2">
					<Button size="sm" onclick={handleSave}>{editingId ? 'Update' : 'Save'}</Button>
					<Button variant="ghost" size="sm" onclick={resetForm}>Cancel</Button>
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
								<p class="text-xs text-muted-foreground mt-1 line-clamp-2">{preset.systemMessage}</p>
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
								variant="ghost"
								size="icon"
								class="h-7 w-7"
								onclick={() => presetsStore.applyPreset(preset.id)}
								title="Apply preset"
							>
								<Play class="h-3.5 w-3.5" />
							</Button>
							<Button
								variant="ghost"
								size="icon"
								class="h-7 w-7"
								onclick={() => startEdit(preset)}
								title="Edit"
							>
								<Pencil class="h-3.5 w-3.5" />
							</Button>
							<Button
								variant="ghost"
								size="icon"
								class="h-7 w-7 text-destructive"
								onclick={() => handleDelete(preset.id)}
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
