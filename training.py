from test import test
from torch.nn import BCEWithLogitsLoss
import torch
import os
import wandb
from argparse import Namespace
import time
from torch.cuda.amp import GradScaler, autocast

def train(model: torch.nn.Module,
          trainloader: torch.nn.Module,
          valloader: torch.nn.Module,
          testloader: torch.nn.Module,
          args: Namespace) -> torch.nn.Module:
    """train model"""
    run = wandb.init(project=args.project, name=args.wandbname, save_code=True,
                     settings=wandb.Settings(start_method='fork'), tags=[args.tag])
    wandb.config.update(args)

    epochs = args.n_epoch
    model.train()
    model = model.cuda()
    if args.n_classes == 1:
        loss_module_instance = torch.nn.BCEWithLogitsLoss()
    else:
        loss_module_instance = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0.000005)

    scaler = GradScaler()

    with torch.no_grad():
        with autocast():
            start_test = time.time()
            metrics = test(model, testloader=testloader)
            end_test = time.time()

        avg_score_higher_test, avg_score_lower_test, auc_value_higher_test, auc_value_lower_test, predictions, _, labels, f1_higher, test_predictions0, test_predictions1 = metrics
        wandb.log({
            "acc_higher_test": avg_score_higher_test, "acc_lower_test": avg_score_lower_test,
            "auc_higher_test": auc_value_higher_test, "f1_higher_test":f1_higher,"epoch": -1, "lr": scheduler.get_last_lr()[0]
        })

    BestPerformance = 0
    # Start training
    print("🟢 Starting training with Automatic Mixed Precision...")
    for epoch in range(epochs):
        start_training = time.time()
        if hasattr(model, "preloop"):
            model.preloop(epoch, trainloader)
        running_loss = 0.0
        batch_count = 0

        for _, data in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            data = data.cuda()
            x, edge_index, childof, level = data.x, data.edge_index, data.childof, data.level
            edge_index2 = data.edge_index_2 if data.__contains__("edge_index_2") else None
            edge_index3 = data.edge_index_3 if data.__contains__("edge_index_3") else None

            with autocast():
                results = model(x, edge_index, level, childof, edge_index2, edge_index3)
                if args.n_classes == 1:
                    bag_label = data.y.float()
                else:
                    bag_label = data.y.long().view(-1)
                loss = model.compute_loss(loss_module_instance, results, bag_label,epoch)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

        end_training = time.time()
        avg_loss = running_loss / batch_count if batch_count > 0 else 0
        wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch})

        scheduler.step()

        if epoch > 15:
            with torch.no_grad():
                with autocast():
                    start_test = time.time()
                    metrics = test(model, testloader=testloader)
                    end_test = time.time()
                avg_score_higher_test, avg_score_lower_test, auc_value_higher_test, auc_value_lower_test, predictions, _, labels, f1_higher, test_predictions0, test_predictions1 = metrics

                wandb.log({
                    "acc_higher_test": avg_score_higher_test, "acc_lower_test": avg_score_lower_test,
                    "auc_higher_test": auc_value_higher_test, "f1_higher_test":f1_higher,
                    "epoch": epoch, "lr": scheduler.get_last_lr()[0]
                })
                print(f"Epoch{epoch}:acc_higher_test = {avg_score_higher_test}, acc_lower_test={avg_score_lower_test}, auc_higher_test = {auc_value_higher_test}\n"
                      f"f1_higher_test={f1_higher}, lr={scheduler.get_last_lr()[0]}")
                performance = float(auc_value_higher_test)
                if performance > BestPerformance:
                    wandb.log({"best accuracy": avg_score_higher_test, "best auc higher": auc_value_higher_test,
                               "best auc lower": auc_value_lower_test,"best f1 higher":f1_higher})
                    print(
                        f"Epoch{epoch}:best accuracy = {avg_score_higher_test}, best auc higher={auc_value_higher_test}, best auc lower = {auc_value_lower_test}, best f1 higher={f1_higher}\n"
                    )
                    BestPerformance = performance
                    model.eval()
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
                    wandb.save(os.path.join(wandb.run.dir, "model.pt"))

        print("training_time: %s  seconds" % (end_training - start_training))
        if 'end_test' in locals():
            print("test_time: %s  seconds" % (end_test - start_test))

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "final.pt"))
    wandb.save(os.path.join(wandb.run.dir, "final.pt"))
    wandb.finish()
    return model

