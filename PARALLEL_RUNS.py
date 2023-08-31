import mlflow

mlflow.projects.run(
    uri=".",
    run_name="e_30_lr_0.003",
    entry_point="train",
    backend='local', 
    synchronous=False,
    parameters={
        'epochs': 30,
        'lr':0.003
    },)

mlflow.projects.run(
    uri=".",
    run_name="e_50_lr_0.001",
    entry_point="train",
    backend='local', 
    synchronous=False,
    parameters={
        'epochs': 50,
        'lr':0.001
    })

mlflow.projects.run(
    uri=".",
    run_name="e_50_lr_0.005",
    entry_point="train",
    backend='local', 
    synchronous=False,
    parameters={
        'epochs': 50,
        'lr':0.005
    })

mlflow.projects.run(
    uri=".",
    run_name="e_50_lr_0.002",
    entry_point="train",
    backend='local', 
    synchronous=False,
    parameters={
        'epochs': 50,
        'lr':0.002
    })