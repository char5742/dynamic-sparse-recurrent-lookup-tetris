using Test
using JLD2
using JSON3
using Lux
using Random
using SHA

if !isdefined(Main, :DistillationDatasetBridge)
    include(joinpath(@__DIR__, "prepare_distillation_dataset.jl"))
end
using .DistillationDatasetBridge
using .PaperHayCell
using .PaperDigitalTwin

file_sha256(path) = bytes2hex(SHA.sha256(read(path)))

function write_canonical_fixture(root; spike_auroc=0.99)
    dataset_root = joinpath(root, "teacher")
    mkpath(dataset_root)
    tree = paper_hay_tree()
    segments = compartment_count(tree)
    detailed_hash = file_sha256(joinpath(@__DIR__, "PaperHayCell.jl"))
    morphology_hash = canonical_morphology_sha256()
    config = TwinConfig(
        segments=segments,
        nmda_regions=4,
        memory_units=12,
        core_dim=8,
        dt_ms=1,
        bank_seed=0x1234,
    )
    samples = 3
    time_steps = 8
    contacts = 4
    contact_segment = Matrix{Int16}(undef, contacts, samples)
    contact_kind = Matrix{UInt8}(undef, contacts, samples)
    contact_strength = fill(0.35f0, contacts, samples)
    event_spike = falses(contacts, time_steps, samples)
    @inbounds for sample in 1:samples
        contact_segment[:, sample] .= Int16[
            tree.basal_terminals[1],
            tree.apical_trunk[2],
            tree.tuft_terminals[1],
            tree.apical_trunk[3],
        ]
        contact_kind[:, sample] .= UInt8[1, 1, 1, 2]
        event_spike[1, 2, sample] = true
        event_spike[2, 3, sample] = true
        event_spike[3, 4, sample] = true
        event_spike[4, 6, sample] = true
    end
    target_voltage = fill(-70.0f0, time_steps, samples)
    target_spike = zeros(Float32, time_steps, samples)
    target_nmda = zeros(Float32, 4, time_steps, samples)
    target_compartment_voltage = Array{Float32,3}(
        undef,
        segments,
        time_steps,
        samples,
    )
    @inbounds for sample in 1:samples, time in 1:time_steps,
        compartment in 1:segments
        target_compartment_voltage[compartment, time, sample] =
            -80.0f0 + 0.1f0 * compartment + 0.01f0 * time
    end
    split_code = UInt8[1, 2, 3]
    metadata = (;
        source_kind="canonical_julia",
        cell_mechanism_sha256=detailed_hash,
        morphology_sha256=morphology_hash,
        compartment_region=copy(tree.region),
        compartment_distance_um=copy(tree.distance_um),
    )
    shard_path = joinpath(dataset_root, "shard.jld2")
    jldsave(
        shard_path;
        twin_config=config,
        contact_segment,
        contact_kind,
        contact_strength,
        event_spike,
        input=nothing,
        target_voltage,
        target_spike,
        target_nmda,
        target_compartment_voltage,
        split_code,
        metadata,
    )
    shard_hash = file_sha256(shard_path)
    manifest = (;
        schema_version=1,
        source_kind="canonical_julia",
        teacher_backend="canonical_julia",
        teacher_hash=detailed_hash,
        cell_mechanism_sha256=detailed_hash,
        morphology_sha256=morphology_hash,
        shards=[(;
            path="shard.jld2",
            sha256=shard_hash,
        )],
    )
    manifest_path = joinpath(dataset_root, "manifest.json")
    open(manifest_path, "w") do stream
        JSON3.pretty(stream, manifest)
    end

    model = build_paper_twin(config; input_density=0.5)
    parameters, _ = Lux.setup(Xoshiro(0x2345), model)
    normalizer = TwinNormalizer(
        zeros(Float32, config.input_dim),
        ones(Float32, config.input_dim),
        0.0f0,
        1.0f0,
        zeros(Float32, config.nmda_regions),
        ones(Float32, config.nmda_regions),
    )
    frozen = freeze_twin(
        model,
        parameters,
        normalizer;
        metadata=(;
            teacher_hash=detailed_hash,
            detailed_teacher_hash=detailed_hash,
            cell_mechanism_sha256=detailed_hash,
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
        manifest_path,
        shard_path,
        twin_path,
        frozen,
        detailed_hash,
        morphology_hash,
        config,
    )
end

@testset "frozen twin to final 11-state dataset bridge" begin
    mktempdir() do directory
        fixture = write_canonical_fixture(directory)
        output = joinpath(directory, "distillation.jld2")
        report = prepare_distillation_dataset(
            DistillationDatasetConfig(
                dataset_path=fixture.dataset_root,
                twin_artifact=fixture.twin_path,
                output_path=output,
                source_kind=:canonical_julia,
                twin_batch_size=2,
                expected_detailed_kernel_sha256=
                    fixture.detailed_hash,
                expected_morphology_sha256=
                    fixture.morphology_hash,
                expected_twin_parameter_sha256=
                    fixture.frozen.parameter_sha256,
                expected_twin_artifact_sha256=
                    fixture.frozen.artifact_sha256,
            ),
        )
        @test report.schema == DISTILLATION_DATASET_SCHEMA
        @test report.samples == 3
        @test report.split_counts == (
            train=1,
            validation=1,
            test=1,
        )
        @test report.mixed_supervision
        @test report.twin_gate_passed
        @test isfile(output)

        payload = JLD2.load(output)["dataset"]
        @test payload.schema == DISTILLATION_DATASET_SCHEMA
        @test size(payload.input) ==
            (fixture.config.input_dim, 8, 3)
        @test size(payload.target_voltage) == (8, 3)
        @test size(payload.target_spike) == (8, 3)
        @test size(payload.target_nmda) == (4, 8, 3)
        @test size(payload.target_calcium_event) == (8, 3)
        @test size(payload.target_dendritic_voltage) == (4, 8, 3)
        @test payload.train_indices == [1]
        @test payload.validation_indices == [2]
        @test payload.test_indices == [3]
        @test payload.mixed_supervision
        @test payload.frozen_twin_parameter_hash ==
            fixture.frozen.parameter_sha256
        @test payload.frozen_twin_artifact_hash ==
            fixture.frozen.artifact_sha256
        @test payload.detailed_kernel_hash == fixture.detailed_hash
        @test payload.morphology_hash == fixture.morphology_hash
        @test length(payload.metadata.input_compartment) ==
            fixture.config.input_dim
        @test length(payload.metadata.input_receptor) ==
            fixture.config.input_dim
        @test length(payload.metadata.input_plane) ==
            fixture.config.input_dim
        @test all(isfinite, payload.target_calcium_event)
        @test all(isfinite, payload.target_dendritic_voltage)

        prediction = twin_forward(fixture.frozen, payload.input)
        @test payload.target_voltage == prediction.voltage
        @test payload.target_spike == prediction.spike_probability
        @test payload.target_nmda == prediction.nmda
        @test payload.metadata.mixed_supervision_provenance.
            target_voltage ==
            "actual frozen PaperDigitalTwin inference"
        @test occursin(
            "detailed",
            payload.metadata.mixed_supervision_provenance.
                target_calcium_event,
        )
    end
end

@testset "bridge rejects hash mismatch and below-gate twin" begin
    mktempdir() do directory
        fixture = write_canonical_fixture(directory)
        @test_throws ErrorException prepare_distillation_dataset(
            DistillationDatasetConfig(
                dataset_path=fixture.dataset_root,
                twin_artifact=fixture.twin_path,
                output_path=joinpath(directory, "bad_hash.jld2"),
                source_kind=:canonical_julia,
                expected_morphology_sha256=repeat("0", 64),
            ),
        )
    end
    mktempdir() do directory
        fixture = write_canonical_fixture(
            directory;
            spike_auroc=0.90,
        )
        @test_throws ErrorException prepare_distillation_dataset(
            DistillationDatasetConfig(
                dataset_path=fixture.dataset_root,
                twin_artifact=fixture.twin_path,
                output_path=joinpath(directory, "below_gate.jld2"),
                source_kind=:canonical_julia,
            ),
        )
    end
end

@testset "bridge rejects source relabelling and shard tampering" begin
    mktempdir() do directory
        fixture = write_canonical_fixture(directory)
        @test_throws ErrorException prepare_distillation_dataset(
            DistillationDatasetConfig(
                dataset_path=fixture.dataset_root,
                twin_artifact=fixture.twin_path,
                output_path=joinpath(directory, "official.jld2"),
                source_kind=:official_neuron,
            ),
        )
        open(fixture.shard_path, "a") do stream
            write(stream, UInt8(0))
        end
        @test_throws ErrorException prepare_distillation_dataset(
            DistillationDatasetConfig(
                dataset_path=fixture.dataset_root,
                twin_artifact=fixture.twin_path,
                output_path=joinpath(directory, "tampered.jld2"),
                source_kind=:canonical_julia,
            ),
        )
    end
end

println("prepare_distillation_dataset tests passed")
