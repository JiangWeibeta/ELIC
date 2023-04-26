import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils.metrics import compute_metrics
from utils.utils import *


def test_one_epoch(epoch, test_dataloader, model, criterion, save_dir, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            if out_criterion["mse_loss"] is not None:
                mse_loss.update(out_criterion["mse_loss"])
            if out_criterion["ms_ssim_loss"] is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

            rec = torch2img(out_net['x_hat'])
            img = torch2img(d)
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)

    if out_criterion["mse_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.4f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch + 1)
    if out_criterion["ms_ssim_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MS-SSIM loss: {ms_ssim_loss.avg:.4f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format('[val]: ms_ssim_loss'), ms_ssim_loss.avg, epoch + 1)

    return loss.avg


def compress_one_image(model, x, stream_path, H, W, img_name):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.compress(x)
    torch.cuda.synchronize()
    end_time = time.time()
    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    enc_time = end_time - start_time
    return bpp, enc_time


def decompress_one_image(model, stream_path, img_name):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.decompress(strings, shape)
    torch.cuda.synchronize()
    end_time = time.time()
    dec_time = end_time - start_time
    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    return x_hat, dec_time



def test_model(test_dataloader, net, logger_test, save_dir, epoch):
    net.eval()
    device = next(net.parameters()).device

    avg_psnr = AverageMeter()
    avg_ms_ssim = AverageMeter()
    avg_bpp = AverageMeter()
    avg_deocde_time = AverageMeter()
    avg_encode_time = AverageMeter()

    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            img = img.to(device)
            B, C, H, W = img.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W

            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            # net.update_resolutions(img_pad.size(2) // 16, img_pad.size(3) // 16)
            bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
            x_hat, dec_time = decompress_one_image(model=net, stream_path=save_dir, img_name=str(i))
            rec = torch2img(x_hat)
            img = torch2img(img)
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            p, m = compute_metrics(rec, img)
            avg_psnr.update(p)
            avg_ms_ssim.update(m)
            avg_bpp.update(bpp)
            avg_deocde_time.update(dec_time)
            avg_encode_time.update(enc_time)
            logger_test.info(
                f"Image[{i}] | "
                f"Bpp loss: {bpp:.4f} | "
                f"PSNR: {p:.4f} | "
                f"MS-SSIM: {m:.4f} | "
                f"Encoding Latency: {enc_time:.4f} | "
                f"Decoding latency: {dec_time:.4f}"
            )
    logger_test.info(
        f"Epoch:[{epoch}] | "
        f"Avg Bpp: {avg_bpp.avg:.7f} | "
        f"Avg PSNR: {avg_psnr.avg:.7f} | "
        f"Avg MS-SSIM: {avg_ms_ssim.avg:.7f} | "
        f"Avg Encoding Latency: {avg_encode_time.avg:.6f} | "
        f"Avg Decoding latency: {avg_deocde_time.avg:.6f}"
    )