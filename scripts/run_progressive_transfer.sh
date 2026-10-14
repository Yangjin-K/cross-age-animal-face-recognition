#!/bin/bash
set -e

# ==============================================================================
# PROGRESSIVE DOMAIN ADAPTATION SCRIPT (Full Pipeline)
# Strategy: Non-human -> CACD (~160k) -> AgeDB (~16k) -> FGNET (~1k)
# ==============================================================================

CONFIG_FILE="configs/config_progressive.yaml"
ORIGINAL_CKPT="fixed_swin_t_projection_for_inference.pth"
CACD_CKPT="backbone_swin_t_cacd_base.pth"
AGEDB_CKPT="backbone_swin_t_agedb_base.pth"
FGNET_CKPT="backbone_swin_t_fgnet_final.pth"

echo "================================================================="
echo "STEP 1: Finetuning original non-human backbone on CACD_dataset (Ultra-Large)"
echo "Goal: Bulk robust transformation from animal to human face domain"
echo "================================================================="
python code/progressive_transfer.py \
    --config ${CONFIG_FILE} \
    --dataset CACD_dataset \
    --pretrained_path ${ORIGINAL_CKPT} \
    --save_path ${CACD_CKPT} \
    --epochs 5 \
    --lr_backbone 5e-5 \
    --lr_head 1e-3

echo "================================================================="
echo "STEP 2: Transferring the CACD model to AgeDB (Medium)"
echo "Goal: Refine human features strictly on balanced cross-age pairs"
echo "================================================================="
python code/progressive_transfer.py \
    --config ${CONFIG_FILE} \
    --dataset AgeDB \
    --pretrained_path ${CACD_CKPT} \
    --save_path ${AGEDB_CKPT} \
    --epochs 20 \
    --lr_backbone 2e-5 \
    --lr_head 5e-4

echo "================================================================="
echo "STEP 3: Transferring the AgeDB model to FGNET (Small/Hard)"
echo "Goal: Mitigate extreme overfitting on FGNET using the highly robust proxy"
echo "================================================================="
python code/progressive_transfer.py \
    --config ${CONFIG_FILE} \
    --dataset FGNET \
    --pretrained_path ${AGEDB_CKPT} \
    --save_path ${FGNET_CKPT} \
    --epochs 50 \
    --lr_backbone 1e-5 \
    --lr_head 5e-4

echo "================================================================="
echo "PROGRESSIVE TRANSFER LEARNING COMPLETE."
echo "Final Checkpoints:"
echo "  CACD Base:  ${CACD_CKPT}"
echo "  AgeDB Base: ${AGEDB_CKPT}"
echo "  FGNET Final: ${FGNET_CKPT}"
echo "================================================================="
