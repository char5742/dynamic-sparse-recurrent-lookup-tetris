# HD-SWSNN-TwinProp official Hay teacher

This path is the detailed-model first stage of:

```text
official Hay L5PC
  -> voltage/spike/NMDA digital twin
  -> 11-state cell distillation
  -> frozen internal cell
  -> HD-SWSNN-TwinProp training
```

It loads the published ModelDB accession 139653 model without modifying that
checkout:

- `morphologies/cell1.asc`
- `models/L5PCbiophys3.hoc`
- `models/L5PCtemplate.hoc`
- the original NMODL channel mechanisms compiled as `libnrnmech.so`

The teacher runs NEURON at fixed `dt=0.025 ms` and records at `1 ms`.
AMPA, NMDA and GABA_A are distinct double-exponential conductances. Their
public TwinProp values are:

| receptor | rise | decay | maximum |
|---|---:|---:|---:|
| AMPA | 0.2 ms | 1.7 ms | 0.4 nS |
| NMDA | 0.29 ms | 43 ms | 0.3 nS |
| GABA_A | 0.2 ms | 8 ms | 0.7 nS |

NMDA uses the local segment voltage and the published Jahr-Stevens magnesium
block coefficient `0.062 mV^-1`. Input axons obey Dale's law, contact
conductances are nonnegative, excitatory contacts drive paired AMPA/NMDA, and
placement spans the official basal/apical tree under the one-E/one-I-contact
per micrometre constraint.

## Run

The prepared environment is:

```text
Windows checkout: C:\tmp\hay_modeldb_139653
WSL checkout:     /mnt/c/tmp/hay_modeldb_139653
NEURON venv:      /opt/hd_swsnn_twinprop_neuron
```

From PowerShell:

```powershell
.\generate_neuron_teacher.ps1 -Preset tiny
.\generate_neuron_teacher.ps1 -Preset smoke `
  -OutputDirectory D:\tetris-paper-plus\runs\hd_swsnn_twinprop\neuron_teacher
```

Directly under WSL:

```bash
source /opt/hd_swsnn_twinprop_neuron/bin/activate
python neuron_hay_teacher.py \
  --modeldb-root /mnt/c/tmp/hay_modeldb_139653 \
  --output /tmp/hd-swsnn-neuron-teacher \
  --preset smoke
```

The production preset exposes the public `49,000 train + 1,000 validation +
2,000 held-out`, `10 s` contract. It is intentionally not the default because
it is a large NEURON generation job. The random-drive axon count is recorded
as a reconstruction choice: the preprint does not publish the authors' exact
teacher-data protocol.

Run the integration test:

```bash
source /opt/hd_swsnn_twinprop_neuron/bin/activate
HAY_MODELDB_ROOT=/mnt/c/tmp/hay_modeldb_139653 \
  python -m unittest test_neuron_hay_teacher.py -v
```

## NPZ contract

`manifest.json` identifies schema
`hd_swsnn_twinprop.neuron_teacher.v1`. It contains the full 642-segment
catalog, fixed diagnostic segment indices, axes, units, reconstruction
choices, and hashes for:

- generator source
- ModelDB commit and tracked tree
- morphology
- HOC biophysics and template
- NMODL sources and compiled library
- complete teacher contract
- every shard

Core per-shard arrays use the same semantic axes as the Julia twin:

```text
target_voltage[T,B]                  soma mV
target_spike[T,B]                    soma threshold crossings
target_nmda[4,T,B]                   region-summed NMDA current, nA
target_compartment_voltage[D,T,B]    selected dendritic voltage, mV
target_compartment_nmda[D,T,B]       selected segment NMDA current, nA
target_dendritic_cai[D,T,B]          intracellular Ca, mM
target_dendritic_ica[D,T,B]          total Ca current density, mA/cm2
target_ca_event[D,T,B]               local Ca-event diagnostic
```

The NMDA sign convention is outward-positive, matching
`PaperHayCell.jl`. Contacts and Poisson events are stored without Python
objects:

```text
contact_axon[C,B]        1-based axon
contact_segment[C,B]     1-based official Hay segment
contact_kind[C,B]        1 excitatory, 2 inhibitory
contact_strength[C,B]    fraction of receptor maximum
event_trial_offset[B+1]  compact-event offsets
event_axon[E]            1-based event axon
event_time_bin[E]        0-based 1-ms bin
```

Tiny and smoke presets also save dense `axon_event_spike[A,T,B]` and
compatibility `event_spike[C,T,B]`.

## Reproduction boundary

This is mechanism-faithful to the public path: official Hay morphology and
active channels, voltage-dependent NMDA, receptor-specific kinetics,
apical/basal placement, voltage, soma-spike, NMDA and calcium recordings.

It is not byte-identical to the authors' unpublished TwinProp generator. The
preprint does not publish its exact random-drive protocol, solver settings,
synapse-location optimizer, or code. Every such choice is separated from
paper-explicit values in the manifest.

## Sources and licensing

- Hay et al., *Models of Neocortical Layer 5b Pyramidal Cells Capturing a
  Wide Range of Dendritic and Perisomatic Active Properties* (2011).
- ModelDB accession 139653:
  <https://github.com/ModelDBRepository/139653>
- TwinProp preprint, *What can a neuron compute?* (2026).

The downloaded ModelDB repository currently contains no top-level LICENSE or
COPYING file. This project therefore does not vendor or rewrite those files;
it loads a separately obtained checkout and records its commit/hash. Users
must follow the source authors' and ModelDB's terms when redistributing the
model or generated artifacts.

## Canonical final generator

`neuron_hay_teacher.py` remains an early official-model control. The accepted
production source is `neuron_hay_teacher_final.py`, with wrapper
`generate_neuron_teacher_final.ps1` and schema
`hd_swsnn_twinprop.neuron_teacher.final.v2`.

The final generator uses exactly 50,000 train and 2,000 held-out 10-second
simulations in its production preset. Poisson input is sampled from a
Gaussian-smoothed piecewise-constant instantaneous rate. Gaussian sigma and
constant-rate window are independently sampled from 10--1000 ms for every
simulation. Strength is uniform in `[0,1)`. Contacts use continuous section
positions and never repeat a one-micrometre slot within either the E or I
population.

The one-micrometre slot is the density/placement coordinate, not an additional
electrical compartment. The official Hay `geom_nseg` discretization has 642
electrical segments. Contacts at different micrometre coordinates inside the
same segment therefore share that segment voltage, and their linear receptor
currents are aggregated at the containing segment. `contact_x` and
`contact_location_slot` preserve the placement constraint; `contact_segment`
is the 1-based 642-segment ID consumed by the final distilled/Tetris runtime.

Shards are atomically saved, independently hashed, process-parallel and
resumable through verified `.done.json` files. Soma voltage, spike and
four-region NMDA targets remain at 1 ms; dendritic voltage/NMDA/Ca diagnostics
have configurable segment count and temporal stride.

The paper does not state the rate-level distribution, whether timescales were
sampled linearly or logarithmically, the random-drive axon count, or its E/I
fraction. Defaults use rate levels `Uniform[0,50) Hz`, continuous uniform
timescale sampling, and an explicitly recorded connectivity interpretation.

There is a public-methods ambiguity: the task protocol says 8,000 total
synapses and parenthetically 4,000 E / 4,000 I, while another section says an
average 20 contacts per input axon; the Step-1 random-twin drive does not state
its own axon/contact count. Production therefore fails closed unless
`-Acknowledge8000ContactInterpretation` is supplied. The acknowledged policy is
400 axons, exactly 20 contacts each and 200 E / 200 I axons, yielding exactly
4,000 E and 4,000 I contacts. The manifest explicitly records this as a
reconstruction interpretation and sets `fully_paper_scale_claim=false`.

```powershell
.\generate_neuron_teacher_final.ps1 -Preset smoke -Workers 2
.\generate_neuron_teacher_final.ps1 -Preset production -Workers 4 `
  -Acknowledge8000ContactInterpretation `
  -OutputDirectory D:\tetris-paper-plus\runs\hd_swsnn_twinprop\neuron_final
```
