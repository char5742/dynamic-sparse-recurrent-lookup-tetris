function paper_worker_final(trainer::PaperTrainer)
    model = trainer.model
    aux = register_paper_trainer_aux!(trainer)
    return PaperWorker(
        make_cell_runtime_final(trainer),
        Optim.zero_parameter_tree(trainer.parameters),
        Point.PackScratch(),
        zeros(Float32, model.blocks),
        fill(false, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Int16, model.workspace_k),
        zeros(Float32, model.node_dim, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, 2model.node_dim),
        zeros(Float32, 2model.node_dim),
        zeros(Float32, model.hidden),
        zeros(Float32, 11, model.blocks),
        zeros(Float32, 11),
        zeros(Float32, size(trainer.input_location_utility)),
        zeros(Float32, size(trainer.recurrent_location_utility)),
        zeros(Float32, size(aux.workspace_location_utility)),
        UInt64(0),
    )
end
