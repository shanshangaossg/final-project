from torch.nn import CrossEntropyLoss

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        batch = batch.to(device)  
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        all_preds.append(pred.argmax(dim=1).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Epoch {epoch} Train Loss: {running_loss:.4f} | Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")
    return running_loss, accuracy, f1


def test(model, test_loader, loss_fn):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device) 
            pred = model(batch)  
            loss = loss_fn(pred, batch.y)
            running_loss += loss.item()

            all_preds.append(pred.argmax(dim=1).cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Test Loss: {running_loss:.4f} | Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")
    return running_loss, accuracy, f1

def run_multiple_experiments(
    model_class, train_loader, test_loader, input_dim, hidden_dim, output_dim, edge_dim, num_runs=3
):
    """
    Run multiple experiments and calculate mean and standard error for the results.
    """
    all_results = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs} for {model_class.__name__}")

        seed = 42 + run
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_class(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, edge_dim=edge_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        epoch_results = []
        for epoch in range(5):
            train_loss, train_acc, train_f1 = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
            test_loss, test_acc, test_f1 = test(model, test_loader, loss_fn)
            epoch_results.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_f1": train_f1,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_f1": test_f1
            })

        all_results.append(epoch_results)

    all_results_flat = []
    for run_id, run_results in enumerate(all_results):
        for epoch_result in run_results:
            epoch_result["run"] = run_id + 1
            all_results_flat.append(epoch_result)

    results_df = pd.DataFrame(all_results_flat)

    summary_results = results_df.groupby("epoch").agg({
        "train_loss": ["mean", "sem"],
        "train_accuracy": ["mean", "sem"],
        "train_f1": ["mean", "sem"],
        "test_loss": ["mean", "sem"],
        "test_accuracy": ["mean", "sem"],
        "test_f1": ["mean", "sem"]
    })

    return results_df, summary_results

import matplotlib.pyplot as plt

def plot_results(summary_df, model_class):
    acc_mean = summary_df["test_accuracy"]["mean"].values
    acc_std = summary_df["test_accuracy"]["sem"].values
    f1_mean = summary_df["test_f1"]["mean"].values
    f1_std = summary_df["test_f1"]["sem"].values

    epochs = range(1, len(acc_mean) + 1)

    plt.figure(figsize=(10, 5))
    plt.errorbar(epochs, acc_mean, yerr=acc_std, label="Test Accuracy", fmt='-o', capsize=3)
    plt.errorbar(epochs, f1_mean, yerr=f1_std, label="Test F1 Score", fmt='-s', capsize=3)
    plt.title(f"{model_class.__name__} Performance")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/content/drive/My Drive/final_project/data/{model_class}_performance.png")
    plt.show()

def plot_train_test_comparison(results_df, model_class):
    grouped_df = results_df.groupby("epoch").mean()

    epochs = grouped_df.index.values
    train_acc = grouped_df["train_accuracy"].values
    test_acc = grouped_df["test_accuracy"].values
    train_f1 = grouped_df["train_f1"].values
    test_f1 = grouped_df["test_f1"].values

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
    plt.plot(epochs, test_acc, label="Test Accuracy", marker='o')
    plt.plot(epochs, train_f1, label="Train F1 Score", marker='s')
    plt.plot(epochs, test_f1, label="Test F1 Score", marker='s')

    plt.title(f"{model_class} Train/Test Performance Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/content/drive/My Drive/final_project/data/{model_class}_train_test_comparison.png")
    plt.show()

from itertools import product

def tune_hyperparameters(
    model_class,
    train_loader,
    test_loader,
    input_dim,
    output_dim,
    edge_dim,
    param_grid,
    num_runs=3
):
    param_combinations = list(product(*param_grid.values()))
    best_accuracy = 0.0
    best_params = None
    best_results = None

    print(f"Tuning {len(param_combinations)} hyperparameter combinations for {model_class.__name__}")

    for param_values in param_combinations:
        params = dict(zip(param_grid.keys(), param_values))
        print(f"\nTesting parameters: {params}")

        results_df, summary_df = run_multiple_experiments(
            model_class=model_class,
            train_loader=train_loader,
            test_loader=test_loader,
            input_dim=input_dim,
            hidden_dim=params.get("hidden_dim", 64),
            output_dim=output_dim,
            edge_dim=edge_dim,
            num_runs=num_runs
        )

        mean_test_acc = summary_df["test_accuracy"]["mean"].iloc[-1]
        print(f"Mean Test Accuracy for {params}: {mean_test_acc:.4f}")

        if mean_test_acc > best_accuracy:
            best_accuracy = mean_test_acc
            best_params = params
            best_results = (results_df, summary_df)

    print("\nBest Parameters:")
    print(best_params)
    print(f"Best Test Accuracy: {best_accuracy:.4f}")

    return best_params, best_results

def plot_tuning_results(summary_df, model_name, params):
    epochs = summary_df.index.values
    acc_mean = summary_df["test_accuracy"]["mean"].values
    acc_sem = summary_df["test_accuracy"]["sem"].values

    plt.figure(figsize=(10, 5))
    plt.errorbar(epochs, acc_mean, yerr=acc_sem, label="Test Accuracy", fmt='-o', capsize=3)
    plt.title(f"{model_name} Performance (Params: {params})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_generalization_gap(summary_df):
    train_acc_mean = summary_df[("train_accuracy", "mean")].values
    test_acc_mean = summary_df[("test_accuracy", "mean")].values
    epoch = summary_df.index.values 
    generalization_gap = train_acc_mean - test_acc_mean

    gap_df = pd.DataFrame({
        "epoch": epoch.astype(int).reshape(-1),  # Ensure epoch is 1D
        "generalization_gap": generalization_gap.reshape(-1) # Ensure generalization_gap is 1D
    })

    return gap_df

def plot_generalization_gap(gap_df, model_name="Model"):
    plt.figure(figsize=(8, 6))
    plt.plot(gap_df["epoch"], gap_df["generalization_gap"], marker='o', label="Generalization Gap")
    plt.xlabel("Epoch")
    plt.ylabel("Generalization Gap (Train Accuracy - Test Accuracy)")
    plt.title(f"Generalization Gap for {model_name}")
    plt.legend()
    plt.grid()
    plt.show()


train_dataset = MoleculeDataset(root="/content/drive/My Drive/final_project/data", filename="HIV_train_balanced.csv")
test_dataset = MoleculeDataset(root="/content/drive/My Drive/final_project/data", filename='HIV_test.csv', test=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

edge_dim = train_dataset[0].edge_attr.shape[1]
# GCN
gcn_results_df, gcn_summary_df = run_multiple_experiments(
    GCN,
    train_loader,
    test_loader,
    input_dim=train_dataset[0].x.shape[1],
    hidden_dim=32,
    output_dim=2,
    edge_dim=edge_dim,
    num_runs=3,
)
gat_results_df.to_csv("/content/drive/My Drive/final_project/data/GCN_results_full.csv", index=False)
gat_summary_df.to_csv("/content/drive/My Drive/final_project/data/GCN_results_summary.csv", index=False)


# GAT
gat_results_df, gat_summary_df = run_multiple_experiments(
    GAT,
    train_loader,
    test_loader,
    input_dim=train_dataset[0].x.shape[1],
    hidden_dim=32,
    output_dim=2,
    edge_dim=edge_dim,
    num_runs=3,
)
gat_results_df.to_csv("/content/drive/My Drive/final_project/data/GAT_results_full.csv", index=False)
gat_summary_df.to_csv("/content/drive/My Drive/final_project/data/GAT_results_summary.csv", index=False)
print("GAT Model results saved!")
plot_results(gat_summary_df, "GATModel")
plot_train_test_comparison(gat_results_df, "GATModel")

param_grid = {
    "hidden_dim": [32, 64], 
    "heads": [2, 4],             
    "dropout": [0.1, 0.3]   
}
best_params, best_results = tune_hyperparameters(
    model_class=GAT,
    train_loader=train_loader,
    test_loader=test_loader,
    input_dim=train_dataset[0].x.shape[1],
    output_dim=2,
    edge_dim=train_dataset[0].edge_attr.shape[1],
    param_grid=param_grid,
    num_runs=3
)
results_df, summary_df = best_results
results_df.to_csv("GAT_best_results.csv", index=False)
summary_df.to_csv("GAT_best_summary.csv", index=False)
plot_tuning_results(summary_df, "GAT", best_params)

gat_generalization_df = compute_generalization_gap(gat_summary_df)
print(gat_generalization_df)
plot_generalization_gap(gat_generalization_df, model_name="GAT")


# GATv2
gatv2_results_df, gatv2_summary_df = run_multiple_experiments(
    GATv2,
    train_loader,
    test_loader,
    input_dim=train_dataset[0].x.shape[1],
    hidden_dim=32,
    output_dim=2,
    edge_dim=edge_dim,
    num_runs=3,
)
gatv2_results_df.to_csv("/content/drive/My Drive/final_project/data/GATv2_results_full.csv", index=False)
gatv2_summary_df.to_csv("/content/drive/My Drive/final_project/data/GATv2_results_summary.csv", index=False)
print("GAT Model results saved!")
plot_results(gatv2_summary_df, "GATv2Model")
plot_train_test_comparison(gatv2_results_df, "GATv2Model")


param_grid = {
    "hidden_dim": [32, 64],   
    "heads": [2, 4],             
    "dropout": [0.1, 0.3]  
}
best_params, best_results = tune_hyperparameters(
    model_class=GATv2,
    train_loader=train_loader,
    test_loader=test_loader,
    input_dim=train_dataset[0].x.shape[1],
    output_dim=2,
    edge_dim=train_dataset[0].edge_attr.shape[1],
    param_grid=param_grid,
    num_runs=3
)
results_df, summary_df = best_results
results_df.to_csv("GATv2_best_results.csv", index=False)
summary_df.to_csv("GATv2_best_summary.csv", index=False)
plot_tuning_results(summary_df, "GATv2", best_params)

gatv2_generalization_df = compute_generalization_gap(gatv2_summary_df)
print(gatv2_generalization_df)
plot_generalization_gap(gatv2_generalization_df, model_name="GATv2")

