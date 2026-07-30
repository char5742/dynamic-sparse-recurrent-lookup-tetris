using Test
using JLD2
using JSON3
using Lux
using NPZ
using Random
using SHA

if !isdefined(Main, :DistillationDatasetBridgeFinal)
    include(joinpath(@__DIR__, "prepare_distillation_dataset_final.jl"))
end
using .DistillationDatasetBridgeFinal
using .PaperDigitalTwin

file_sha256(path) = bytes2hex(SHA.sha256(read(path)))

function write_official_fixture(root; spike_auroc=0.99)
    dataset_root = joinpath(root, "official_teacher")
    mkpath(dataset_root)
    segments = [
        (index=1, region_name="soma", distance_um=0.0),
        (index=2, region_name="basal", distance_um=50.0),
        (index=3, region_name="basal", distance_um=200.0),
        (index=4, region_name="apical_trunk", distance_um=250.0),
        (index=5, region_name="apical_trunk", distance_um=780.0),
        (index=6, region_name="apical_tuft", distance_um=800.0),
        (index=7, region_name="apical_tuft", distance_um=1100.0),
        (index=8, region_name="axon", distance_um=20.0),
    ]
    diagnostic = Int32[2, 3, 4, 5, 6, 7]
    trials = 3
    time_steps = 7
    contacts = 5
    contact_axon = Int32[
        1 1 1;
        1 1 1;
        2 2 2;
        3 3 3;
        4 4 4;
    ]
    contact_segment = Int32[
        2 2 2;
        3 3 3;
        4 4 4;
        5 5 5;
        7 7 7;
    ]
    contact_kind = UInt8[
        1 1 1;
        1 1 1;
        1 1 1;
        2 2 2;
        2 2 2;
    ]
    contact_strength = fill(0.3f0, contacts, trials)
    event_trial_offset = Int64[0, 3, 6, 9]
    event_axon = Int32[1, 2, 4, 1, 3, 4, 1, 2, 3]
    event_time_bin = Int32[0, 2, 5, 1, 3, 6, 0, 4, 5]
    target_voltage = fill(-68.0f0, time_steps, trials)
    target_spike = zeros(Float32, time_steps, trials)
    target_nmda = zeros(Float32, 4, time_steps, trials)
    target_compartment_voltage =
        Array{Float32,3}(undef, length(diagnostic), time_steps, trials)
    target_compartment_nmda =
        zeros(Float32, length(diagnostic), time_steps, trials)
    target_dendritic_cai =
        fill(1.0f-4, length(diagnostic), time_steps, trials)
    target_dendritic_ica =
        zeros(Float32, length(diagnostic), time_steps, trials)
    target_ca_event =
        zeros(UInt8, length(diagnostic), time_steps, trials)
    target_ca_event[4, 4, 1] = 1
    @inbounds for trial in 1:trials, time in 1:time_steps,
        row in eachindex(diagnostic)
        target_compartment_voltage[row, time, trial] =
            -75.0f0 + 0.2f0 * row + 0.01f0 * time
    end
    shard_path = joinpath(dataset_root, "teacher_00001.npz")
    NPZ.npzwrite(
        shard_path,
        Dict(
            "sample_indices" => Int32[101, 201, 301],
            "split_code" => UInt8[1, 2, 3],
            "contact_axon" => contact_axon,
            "contact_segment" => contact_segment,
            "contact_kind" => contact_kind,
            "contact_strength" => contact_strength,
            "event_trial_offset" => event_trial_offset,
            "event_axon" => event_axon,
            "event_time_bin" => event_time_bin,
            "diagnostic_segment_indices" => diagnostic,
            "target_voltage" => target_voltage,
            "target_spike" => target_spike,
            "target_nmda" => target_nmda,
            "target_compartment_voltage" =>
                target_compartment_voltage,
            "target_compartment_nmda" => target_compartment_nmda,
            "target_dendritic_cai" => target_dendritic_cai,
            "target_dendritic_ica" => target_dendritic_ica,
            "target_ca_event" => target_ca_event,
        ),
    )
    shard_hash = file_sha256(shard_path)
    modeldb_hash = repeat("1", 64)
    morphology_hash = repeat("2", 64)
    detailed_kernel_hash = repeat("3", 64)
    detailed_teacher_hash = repeat("4", 64)
    manifest = (;
        schema_version=1,
        schema_name=OFFICIAL_NEURON_SCHEMA,
        completion_state="complete",
        modeldb_source_modified_by_generator=false,
        teacher_contract_sha256=detailed_teacher_hash,
        total_segments=length(segments),
        diagnostic_segment_indices=diagnostic,
        regions=(
            "soma",
            "basal",
            "apical_trunk",
            "apical_tuft",
        ),
        segments,
        source_hashes=(;
            modeldb_tree_sha256=modeldb_hash,
            morphology_sha256=morphology_hash,
            mechanism_library_sha256=detailed_kernel_hash,
        ),
        shards=[(;
            path=basename(shard_path),
            sha256=shard_hash,
        )],
    )
    manifest_path = joinpath(dataset_root, "manifest.json")
    open(manifest_path, "w") do stream
        JSON3.pretty(stream, manifest)
    end

    twin_config = TwinConfig(
        segments=length(segments),
        nmda_regions=4,
        memory_units=10,
        core_dim=8,
        dt_ms=1,
        bank_seed=0x1234,
    )
    model = build_paper_twin(twin_config; input_density=0.5)
    parameters, _ = Lux.setup(Xoshiro(0x4567), model)
    normalizer = TwinNormalizer(
        zeros(Float32, twin_config.input_dim),
        ones(Float32, twin_config.input_dim),
        0.0f0,
        1.0f0,
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    frozen = freeze_twin(
        model,
        parameters,
        normalizer;
        metadata=(;
            detailed_teacher_hash,
            teacher_hash=detailed_teacher_hash,
            cell_mechanism_sha256=detailed_kernel_hash,
            morphology_sha256=morphology_hash,
            held_out_test=(;
                spike_auroc,
                voltage_rmse=1.0,
                nmda_normalized_mse=0.1,
            ),
        ),
    )
    twin_path = joinpath(root, "frozen_twin.jld2")
    save_frozen_twin(twin_path, frozen)
    return (;
        dataset_root,
        shard_path,
        manifest_path,
        twin_path,
        frozen,
        twin_config,
        detailed_teacher_hash,
        detailed_kernel_hash,
        morphology_hash,
        modeldb_hash,
    )
end

@testset "official detailed -> frozen twin -> 11-state dataset" begin
    mktempdir() do directory
        fixture = write_official_fixture(directory)
        output = joinpath(directory, "prepared.jld2")
        report = prepare_distillation_dataset(
            PrepareDistillationConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_path=output,
                source_kind=:official_neuron,
                twin_batch_size=2,
                expected_detailed_teacher_sha256=
                    fixture.detailed_teacher_hash,
                expected_detailed_kernel_sha256=
                    fixture.detailed_kernel_hash,
                expected_morphology_sha256=fixture.morphology_hash,
                expected_modeldb_source_sha256=fixture.modeldb_hash,
                expected_twin_parameter_sha256=
                    fixture.frozen.parameter_sha256,
                expected_twin_artifact_sha256=
                    fixture.frozen.artifact_sha256,
            ),
        )
        @test report.schema == PREPARED_DATASET_SCHEMA
        @test report.samples == 3
        @test report.split_counts == (
            train=1,
            validation=1,
            test=1,
        )
        @test report.frozen_max_delta_before == 0
        @test report.frozen_max_delta_after == 0
        payload = JLD2.load(output)["dataset"]
        @test payload.schema == PREPARED_DATASET_SCHEMA
        @test payload.official_neuron_schema == OFFICIAL_NEURON_SCHEMA
        @test payload.digital_twin_gate_passed
        @test payload.mixed_supervision
        @test payload.source_sample_indices == Int32[101, 201, 301]
        @test size(payload.input) ==
            (fixture.twin_config.input_dim, 7, 3)
        @test size(payload.target_voltage) == (7, 3)
        @test size(payload.target_spike) == (7, 3)
        @test size(payload.target_spike_logit) == (7, 3)
        @test size(payload.target_nmda) == (4, 7, 3)
        @test size(payload.target_calcium_event) == (7, 3)
        @test size(payload.target_dendritic_voltage) == (4, 7, 3)
        @test payload.target_calcium_event[4, 1] == 1
        @test payload.train_indices == [1]
        @test payload.validation_indices == [2]
        @test payload.test_indices == [3]
        @test payload.teacher_hash == fixture.detailed_teacher_hash
        @test payload.frozen_twin_artifact_hash ==
            fixture.frozen.artifact_sha256
        @test payload.metadata.input_compartment ==
            repeat(collect(1:8), 6)
        @test length(payload.metadata.input_receptor) ==
            fixture.twin_config.input_dim

        prediction = twin_forward(fixture.frozen, payload.input)
        @test payload.target_voltage == prediction.voltage
        @test payload.target_spike == prediction.spike_probability
        @test payload.target_spike_logit == prediction.spike_logit
        @test payload.target_nmda == prediction.nmda
    end
end

@testset "lineage, source schema and twin gate are fail-closed" begin
    mktempdir() do directory
        fixture = write_official_fixture(directory)
        @test_throws ErrorException prepare_distillation_dataset(
            PrepareDistillationConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_path=joinpath(directory, "bad_hash.jld2"),
                source_kind=:official_neuron,
                expected_morphology_sha256=repeat("f", 64),
            ),
        )
    end
    mktempdir() do directory
        fixture = write_official_fixture(directory; spike_auroc=0.9)
        @test_throws ErrorException prepare_distillation_dataset(
            PrepareDistillationConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_path=joinpath(directory, "below_gate.jld2"),
                source_kind=:official_neuron,
            ),
        )
    end
    mktempdir() do directory
        fixture = write_official_fixture(directory)
        @test_throws ErrorException prepare_distillation_dataset(
            PrepareDistillationConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_path=joinpath(directory, "relabeled.jld2"),
                source_kind=:canonical_julia,
            ),
        )
        open(fixture.shard_path, "a") do stream
            write(stream, UInt8(0))
        end
        @test_throws ErrorException prepare_distillation_dataset(
            PrepareDistillationConfig(
                dataset_path=fixture.dataset_root,
                frozen_twin_path=fixture.twin_path,
                output_path=joinpath(directory, "tampered.jld2"),
                source_kind=:official_neuron,
            ),
        )
    end
end

println("prepare_distillation_dataset_final tests passed")
