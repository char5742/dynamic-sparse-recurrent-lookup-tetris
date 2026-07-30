using Test
using JLD2
using JSON3
using Lux
using NPZ
using Random
using SHA

include(joinpath(
    @__DIR__,
    "prepare_distillation_dataset_release_canonical.jl",
))

const ReleaseBridge =
    Main.DistillationDatasetBridgeReleaseCanonical
const ReleaseV6 = ReleaseBridge.V6
const TwinRuntime = Main.PaperDigitalTwin

release_file_sha256(path) =
    bytes2hex(SHA.sha256(read(path)))

function release_write_json(path, value)
    open(path, "w") do stream
        JSON3.pretty(stream, value)
        write(stream, '\n')
    end
    return path
end

function release_segments()
    specifications = (
        ("soma", 0, 0.0),
        ("basal", 1, 50.0),
        ("basal", 1, 200.0),
        ("apical_trunk", 2, 250.0),
        ("apical_trunk", 2, 780.0),
        ("apical_tuft", 3, 800.0),
        ("apical_tuft", 3, 1100.0),
        ("soma", 0, 20.0),
    )
    return [
        (;
            index,
            section_name="section_$index",
            section_region=region,
            x=0.5,
            distance_um=distance,
            length_um=20.0,
            diameter_um=1.0,
            area_um2=20.0,
            region_code=code,
            region_name=region,
            has_calcium=region in (
                "apical_trunk",
                "apical_tuft",
            ),
        )
        for (index, (region, code, distance)) in
            enumerate(specifications)
    ]
end

function release_base_arrays()
    samples = 4
    time_steps = 7
    diagnostic = Int32[2, 3, 4, 5, 6, 7]
    diagnostic_time = Int32[0, 2, 4, 6]
    axons = 4
    contact_trial_offset = Int64[0, 3, 6, 9, 12]
    contact_axon = Int32[
        1, 2, 4,
        1, 3, 4,
        1, 2, 3,
        2, 3, 4,
    ]
    contact_segment = Int32[
        2, 5, 7,
        3, 4, 6,
        2, 5, 7,
        3, 4, 6,
    ]
    contact_kind = UInt8[
        1, 1, 2,
        1, 2, 2,
        1, 1, 2,
        1, 2, 2,
    ]
    contact_strength = Float32[
        0.2, 0.4, 0.3,
        0.3, 0.2, 0.4,
        0.5, 0.2, 0.3,
        0.4, 0.3, 0.2,
    ]
    event_trial_offset = Int64[0, 3, 6, 9, 12]
    # Intentionally axon-major/non-time-major within each trial.
    event_axon = Int32[
        1, 2, 1,
        4, 1, 3,
        2, 1, 3,
        4, 2, 3,
    ]
    event_time_bin = Int32[
        5, 1, 0,
        6, 2, 4,
        5, 0, 3,
        6, 1, 4,
    ]
    event_count = UInt8[
        1, 2, 1,
        1, 1, 2,
        1, 1, 1,
        2, 1, 1,
    ]
    diagnostic_shape =
        (length(diagnostic), length(diagnostic_time), samples)
    target_compartment_voltage =
        Array{Float32,3}(undef, diagnostic_shape)
    for trial in 1:samples,
        time in 1:length(diagnostic_time),
        segment in 1:length(diagnostic)
        target_compartment_voltage[segment, time, trial] =
            -72.0f0 + 0.2f0 * segment +
            0.03f0 * time + 0.01f0 * trial
    end
    target_ca_event = zeros(UInt8, diagnostic_shape)
    target_ca_event[4, 2, 1] = 1
    target_ca_event[5, 3, 4] = 1
    return Dict{String,Any}(
        "sample_indices" => Int32[11, 22, 33, 44],
        "split_code" => UInt8[1, 1, 1, 3],
        "axon_kind" => repeat(
            reshape(UInt8[1, 1, 2, 2], axons, 1),
            1,
            samples,
        ),
        "contact_count_per_axon" =>
            fill(Int32(1), axons, samples),
        "contact_trial_offset" => contact_trial_offset,
        "contact_axon" => contact_axon,
        "contact_segment" => contact_segment,
        "contact_location_slot" =>
            Int32.(collect(1:length(contact_axon))),
        "contact_section" => copy(contact_segment),
        "contact_x" =>
            Float32.(range(0.1, 0.9; length=length(contact_axon))),
        "contact_path_distance_um" =>
            20.0f0 .* Float32.(collect(1:length(contact_axon))),
        "contact_kind" => contact_kind,
        "contact_strength" => contact_strength,
        "event_trial_offset" => event_trial_offset,
        "event_axon" => event_axon,
        "event_time_bin" => event_time_bin,
        "event_count" => event_count,
        "rate_window_ms" => fill(20.0f0, samples),
        "rate_sigma_ms" => fill(15.0f0, samples),
        "rate_mean_hz" => fill(10.0f0, samples),
        "rate_std_hz" => fill(2.0f0, samples),
        "diagnostic_segment_indices" => diagnostic,
        "diagnostic_time_indices" => diagnostic_time,
        "time_ms" => Float32.(1:time_steps),
        "diagnostic_time_ms" =>
            Float32.(diagnostic_time .+ 1),
        "target_voltage" =>
            fill(-68.0f0, time_steps, samples),
        "target_spike" =>
            zeros(Float32, time_steps, samples),
        "target_nmda" =>
            zeros(Float32, 4, time_steps, samples),
        "target_compartment_voltage" =>
            target_compartment_voltage,
        "target_compartment_nmda" =>
            zeros(Float32, diagnostic_shape),
        "target_dendritic_cai" =>
            fill(1.0f-4, diagnostic_shape),
        "target_dendritic_ica" =>
            zeros(Float32, diagnostic_shape),
        "target_ca_event" => target_ca_event,
    )
end

function release_contract(config, source_hashes)
    without_hash = (;
        schema_name=ReleaseBridge.FINAL_NEURON_SCHEMA,
        model_name="HD-SWSNN-TwinProp",
        config,
        source_hashes,
        location_slot_sha256=repeat("9", 64),
        paper_protocol=(;
            train_simulations=50_000,
            held_out_test_simulations=2_000,
            duration_ms=10_000,
            mean_contacts_per_axon=20,
        ),
    )
    canonical = ReleaseV6._canonical_json(without_hash)
    digest = bytes2hex(SHA.sha256(codeunits(canonical)))
    return merge(
        without_hash,
        (; teacher_contract_sha256=digest),
    ), canonical, digest
end

function release_write_fixture(
    root;
    spike_mode::Symbol=:faithful,
)
    dataset_root = joinpath(root, "official_final_v2")
    mkpath(dataset_root)
    arrays = release_base_arrays()
    shard_path =
        joinpath(dataset_root, "neuron_hay_final_00001.npz")
    NPZ.npzwrite(shard_path, arrays)
    source_config = (;
        preset="fixture",
        train_trials=3,
        test_trials=1,
        duration_ms=7,
        axons=4,
        mean_contacts_per_axon=3.0,
        diagnostic_segments=6,
        diagnostic_stride_bins=2,
        shard_size=4,
        dt_ms=0.025,
        sample_dt_ms=1.0,
    )
    morphology_hash = repeat("2", 64)
    kernel_hash = repeat("3", 64)
    modeldb_hash = repeat("4", 64)
    source_hashes = (;
        modeldb_tree_sha256=modeldb_hash,
        morphology_sha256=morphology_hash,
        mechanism_library_sha256=kernel_hash,
        final_generator_source_sha256=repeat("5", 64),
    )
    contract, canonical, contract_hash =
        release_contract(source_config, source_hashes)
    segments = release_segments()
    shard_record = (;
        schema_name=ReleaseBridge.FINAL_NEURON_SCHEMA,
        teacher_contract_sha256=contract_hash,
        shard_index=1,
        global_first=1,
        global_last=4,
        samples=4,
        contacts=length(arrays["contact_axon"]),
        events=length(arrays["event_axon"]),
        spike_positive_bins=0,
        voltage_minimum_mv=-68.0,
        voltage_maximum_mv=-68.0,
        nmda_absolute_mean_na=0.0,
        path=basename(shard_path),
        sha256=release_file_sha256(shard_path),
        bytes=filesize(shard_path),
        split_counts=(train=3, held_out_test=1),
    )
    manifest = (;
        schema_version=2,
        schema_name=ReleaseBridge.FINAL_NEURON_SCHEMA,
        model_name="HD-SWSNN-TwinProp",
        stage="official_hay_neuron_teacher_final",
        completion_state="complete",
        teacher_contract_sha256=contract_hash,
        teacher_contract=contract,
        teacher_contract_canonical_json=canonical,
        source_hashes,
        modeldb_source_modified_by_generator=false,
        neuron_version="fixture",
        config=source_config,
        validation_from_train_indices=Int[],
        paper_production_contract=(;
            train_trials=50_000,
            held_out_test_trials=2_000,
            duration_ms=10_000,
        ),
        total_segments=length(segments),
        diagnostic_segment_indices=
            arrays["diagnostic_segment_indices"],
        diagnostic_time_indices=
            arrays["diagnostic_time_indices"],
        segments,
        shards=[shard_record],
        completed_trials=4,
        resumable_sidecars_verified=true,
    )
    manifest_path = joinpath(dataset_root, "manifest.json")
    release_write_json(manifest_path, manifest)

    twin_config = TwinRuntime.TwinConfig(
        segments=length(segments),
        nmda_regions=4,
        memory_units=8,
        core_dim=8,
        dt_ms=1,
        bank_seed=0x1234,
    )
    model = TwinRuntime.build_paper_twin(
        twin_config;
        input_density=0.4,
    )
    parameters, _ = Lux.setup(Xoshiro(0x4567), model)
    normalizer = TwinRuntime.TwinNormalizer(
        zeros(Float32, twin_config.input_dim),
        ones(Float32, twin_config.input_dim),
        0.0f0,
        1.0f0,
        zeros(Float32, 4),
        ones(Float32, 4),
    )
    frozen = TwinRuntime.freeze_twin(
        model,
        parameters,
        normalizer;
        metadata=(;
            detailed_teacher_hash=contract_hash,
            teacher_hash=contract_hash,
            cell_mechanism_sha256=kernel_hash,
            morphology_sha256=morphology_hash,
            held_out_test=(; spike_auroc=0.999),
        ),
    )
    twin_path = joinpath(root, "frozen_twin.jld2")
    TwinRuntime.save_frozen_twin(twin_path, frozen)

    source_config_bridge =
        ReleaseBridge.ReleaseStreamingPrepareConfig(
            dataset_path=dataset_root,
            frozen_twin_path=twin_path,
            output_directory=joinpath(root, "unused"),
            validation_samples=1,
            require_full_public_counts=false,
        )
    source, _, _ =
        ReleaseBridge.Production.OrderedBridge.FinalBridge.
        _load_release_source(source_config_bridge)
    shard = ReleaseV6._read_final_shard(shard_path)
    predictions = NamedTuple[]
    for trial in 1:4
        sparse =
            ReleaseV6._sparse_sample(shard, trial, twin_config)
        push!(
            predictions,
            ReleaseV6._infer_sample(
                frozen,
                sparse,
                7,
                3,
            ),
        )
    end
    for trial in 1:4
        arrays["target_voltage"][:, trial] .=
            predictions[trial].voltage
        arrays["target_nmda"][:, :, trial] .=
            predictions[trial].nmda
        order = sortperm(predictions[trial].spike)
        arrays["target_spike"][:, trial] .= 0
        positive = order[(end - 2):end]
        if spike_mode === :faithful
            arrays["target_spike"][positive, trial] .= 1
        elseif spike_mode === :inverse
            arrays["target_spike"][order[1:3], trial] .= 1
        else
            error("unknown spike_mode")
        end
    end
    NPZ.npzwrite(shard_path, arrays)
    manifest_dictionary =
        JSON3.read(read(manifest_path, String), Dict{String,Any})
    manifest_dictionary["shards"][1]["sha256"] =
        release_file_sha256(shard_path)
    manifest_dictionary["shards"][1]["bytes"] =
        filesize(shard_path)
    manifest_dictionary["shards"][1]["spike_positive_bins"] =
        count(>=(0.5), arrays["target_spike"])
    release_write_json(manifest_path, manifest_dictionary)
    return (;
        dataset_root,
        manifest_path,
        shard_path,
        twin_path,
        frozen,
        twin_config,
        contract_hash,
        predictions,
    )
end

function release_assert_event_order(dataset)
    @test dataset.event_order == "time_then_axon"
    for trial in 1:(length(dataset.event_trial_offset) - 1)
        first_event = Int(dataset.event_trial_offset[trial]) + 1
        last_event = Int(dataset.event_trial_offset[trial + 1])
        first_event > last_event && continue
        @test issorted(collect(zip(
            dataset.event_time_bin[first_event:last_event],
            dataset.event_axon[first_event:last_event],
        )))
    end
end

@testset "canonical final-v2 release bridge" begin
    mktempdir() do directory
        fixture = release_write_fixture(directory)
        output = joinpath(directory, "release_shards")
        config = ReleaseBridge.ReleaseStreamingPrepareConfig(
            dataset_path=fixture.dataset_root,
            frozen_twin_path=fixture.twin_path,
            output_directory=output,
            validation_samples=1,
            time_chunk=3,
            output_shard_samples=2,
            minimum_twin_spike_auroc=0.985,
            auroc_histogram_bins=1024,
            require_full_public_counts=false,
        )
        report =
            ReleaseBridge.prepare_distillation_dataset_release(config)
        @test report.schema == ReleaseBridge.RELEASE_DATASET_SCHEMA
        @test report.total_samples == 4
        @test report.split_counts ==
            (train=2, validation=1, test=1)
        @test report.digital_twin_gate_passed
        @test report.recomputed_twin_gate.spike_auroc == 1.0
        @test report.frozen_max_delta_before == 0
        @test report.frozen_max_delta_after == 0
        @test !report.dense_memory_scales_with_total_samples
        @test report.peak_dense_chunk_bytes ==
            fixture.twin_config.input_dim * 3 * sizeof(Float32)

        manifest = JSON3.read(read(report.manifest_path, String))
        @test manifest.official_neuron_schema ==
            ReleaseBridge.FINAL_NEURON_SCHEMA
        @test manifest.completion_state == "complete"
        @test !manifest.promotion_eligible
        @test manifest.event_order == "time_then_axon"
        @test manifest.source_public_counts.train_pool == 50_000
        @test manifest.source_public_counts.held_out_test == 2_000
        @test manifest.validation_derivation.selected == 1
        @test manifest.split_counts.train == 2
        @test manifest.split_counts.validation == 1
        @test manifest.split_counts.test == 1
        @test length(manifest.train_indices) == 2
        @test length(manifest.validation_indices) == 1
        @test length(manifest.test_indices) == 1
        @test length(String(manifest.segment_catalog_sha256)) == 64
        @test String(manifest.frozen_twin_file_sha256) ==
            release_file_sha256(fixture.twin_path)
        @test length(manifest.diagnostic_time_indices) == 4
        @test manifest.target_schema.target_calcium_event.axes ==
            ["diagnostic_time", "trial"]

        source_by_id = Dict(
            Int32(11 * index) => fixture.predictions[index]
            for index in 1:4
        )
        saw_event_multiplicity = false
        for record in manifest.shards
            shard_path = joinpath(output, String(record.path))
            @test release_file_sha256(shard_path) ==
                String(record.sha256)
            dataset = JLD2.load(shard_path)["dataset"]
            @test dataset.schema == ReleaseBridge.RELEASE_SHARD_SCHEMA
            @test dataset.input === nothing
            @test dataset.digital_twin_gate_passed
            @test size(dataset.target_voltage, 1) == 7
            @test size(dataset.target_nmda, 1) == 4
            @test size(dataset.target_calcium_event, 1) == 4
            @test size(dataset.target_dendritic_voltage)[1:2] ==
                (4, 4)
            @test dataset.diagnostic_time_indices ==
                Int32[0, 2, 4, 6]
            saw_event_multiplicity |=
                any(>(UInt8(1)), dataset.event_count)
            release_assert_event_order(dataset)
            for trial in axes(dataset.target_voltage, 2)
                source_prediction =
                    source_by_id[dataset.source_sample_indices[trial]]
                @test dataset.target_voltage[:, trial] ==
                    source_prediction.voltage
                @test dataset.target_spike[:, trial] ==
                    source_prediction.spike
                @test dataset.target_spike_logit[:, trial] ==
                    source_prediction.spike_logit
                @test dataset.target_nmda[:, :, trial] ==
                    source_prediction.nmda
                if dataset.split_code[trial] == UInt8(2)
                    @test dataset.source_split_code[trial] == UInt8(1)
                end
            end
        end
        @test saw_event_multiplicity
    end
end

@testset "gate, schema, raw contract and production counts fail closed" begin
    mktempdir() do directory
        fixture = release_write_fixture(
            directory;
            spike_mode=:inverse,
        )
        output = joinpath(directory, "must_not_publish")
        config = ReleaseBridge.ReleaseStreamingPrepareConfig(
            dataset_path=fixture.dataset_root,
            frozen_twin_path=fixture.twin_path,
            output_directory=output,
            validation_samples=1,
            time_chunk=3,
            output_shard_samples=2,
            minimum_twin_spike_auroc=0.985,
            auroc_histogram_bins=1024,
            require_full_public_counts=false,
        )
        @test_throws ErrorException(
            ReleaseBridge.prepare_distillation_dataset_release(config)
        )
        @test !ispath(output)
        @test isempty(filter(
            name -> startswith(
                name,
                basename(output) * ".staging.",
            ),
            readdir(dirname(output)),
        ))

        strict_output = joinpath(directory, "strict_must_not_publish")
        strict = ReleaseBridge.ReleaseStreamingPrepareConfig(
            dataset_path=fixture.dataset_root,
            frozen_twin_path=fixture.twin_path,
            output_directory=strict_output,
            validation_samples=1,
            require_full_public_counts=true,
        )
        @test_throws ErrorException(
            ReleaseBridge.prepare_distillation_dataset_release(strict)
        )
        @test !ispath(strict_output)

        manifest = JSON3.read(
            read(fixture.manifest_path, String),
            Dict{String,Any},
        )
        original_canonical =
            manifest["teacher_contract_canonical_json"]
        manifest["teacher_contract_canonical_json"] =
            original_canonical * " "
        release_write_json(fixture.manifest_path, manifest)
        @test_throws ErrorException(
            ReleaseBridge.Production.OrderedBridge.FinalBridge.
            _load_release_source(config)
        )
        manifest["teacher_contract_canonical_json"] =
            original_canonical
        manifest["schema_name"] =
            "hd_swsnn_twinprop.neuron_teacher.v1"
        release_write_json(fixture.manifest_path, manifest)
        @test_throws ErrorException(
            ReleaseBridge.Production.OrderedBridge.FinalBridge.
            _load_release_source(config)
        )
    end
end

println("canonical final-v2 release bridge tests passed")
