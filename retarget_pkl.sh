#!/bin/bash
# Retarget pen capping sequences for Inspire Hand

# cap_*_r sequences: bimanual, run both right and left (31 trajectories)
for idx in \
    m_130824 m_130844 m_130919 m_130940 m_130957 m_131018 \
    m_131033 m_131545 m_131600 m_131634 m_131647 m_131703 \
    m_131719 m_131052 m_131108 m_131124 m_131139 m_131154 \
    m_131207 m_131404 m_131419 m_131434 m_131449 m_131503 \
    m_131518 m_131227 m_131241 m_131256 m_131312 m_131328 \
    m_131343; do
    echo "=== Retargeting $idx (right) ==="
    python main/dataset/mano2dexhand.py --data_idx $idx --side right --dexhand inspire --headless --iter 7000
    echo "=== Retargeting $idx (left) ==="
    python main/dataset/mano2dexhand.py --data_idx $idx --side left --dexhand inspire --headless --iter 7000
done

# lh, 9c66f@0 a4082@0
# rh, 897d8@0
