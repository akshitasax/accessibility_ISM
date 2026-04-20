import supremo_lite as sl
#from supremo_lite.mock_models import TestModel, TestModel2D, TORCH_AVAILABLE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyfaidx import Fasta
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
import chrombpnet.training.utils.losses as losses
import chrombpnet.training.utils.one_hot as one_hot
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model

from pyfaidx import Fasta

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ISM Prediction Script")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the dir where ref genome and vcf files live"
    )

    parser.add_argument(
        "--vcf",
        type=str,
        required=True,
        help="Path to the variants VCF file relative to input_dir"
    )

    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Path to the reference FASTA file"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model .h5 file"
    )

    parser.add_argument(
        '--outdir', 
        type=str, 
        default='output', 
        help='Directory to save files'
    )

    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = args.input_dir
    outdir = args.outdir
    out_path = os.path.join(input_dir, outdir)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # ---------------Load data------------------
    reference_path = os.path.join(input_dir, args.ref)
    reference = Fasta(reference_path)

    # make this argpparse
    print(f"\Loading variants from vcf")
    path_to_vcf = os.path.join(input_dir, args.vcf)
    variants = sl.read_vcf(path_to_vcf)
    vcf_name = os.path.splitext(os.path.basename(path_to_vcf))[0]

    print(f"\nLoaded {len(variants)} variants from vcf")

    num_variants = len(variants)

    # Define the file path (relative path to the current working directory)
    variants_path = os.path.join(out_path, f'{vcf_name}_variants.csv')
    # Save the DataFrame to the specified path
    variants.to_csv(variants_path, index=False)

    # ----------------------Generate sequences around variants-----------------------
    # 2114 for chrombpnet
    seq_len = 2114

    # Note: get_alt_ref_sequences is a generator that yields chunks
    print(f"\Generating ref and alt sequences from variants")
    results = list(
        sl.get_alt_ref_sequences(
            reference_fn=reference,
            variants_fn=variants,
            seq_len=seq_len,
            encode=True,  # Get encoded tensors for models
        )
    )

    # Unpack from the first chunk
    alt_seqs, ref_seqs, metadata = results[0]
    
    # Define the file path (relative path to the current working directory)
    metadata_path = os.path.join(out_path, f'{vcf_name}_metadata.csv')
    # Save the DataFrame to the specified path
    metadata.to_csv(metadata_path, index=False)

    print(f"Generated sequences:")
    print(f"  Reference sequences shape: {ref_seqs.shape}")
    print(f"  Alternate sequences shape: {alt_seqs.shape}")
    print(f"  Number of variants: {len(metadata)}")


    # ---------------get predictions from chrombpnet model----------------
    def load_model_wrapper(model_h5):
        # read .h5 model
        custom_objects={"multinomial_nll":losses.multinomial_nll, "tf": tf}    
        get_custom_objects().update(custom_objects)    
        model=load_model(model_h5, compile=False)
        #model.summary()
        return model

    model_path = args.model
    model = load_model_wrapper(model_path)

    # reshaping seqs output from supremo-lite to be what chrombpnet expects
    ref_seqs = ref_seqs.reshape(num_variants, 2114, 4)
    alt_seqs = alt_seqs.reshape(num_variants, 2114, 4)

    ref_pred_logits_wo_bias, ref_pred_logcts_wo_bias = model.predict(ref_seqs)
    alt_pred_logits_wo_bias, alt_pred_logcts_wo_bias = model.predict(alt_seqs)

    def softmax(x, temp=1):
        norm_x = x - np.mean(x,axis=1, keepdims=True)
        return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)
        
    # understand whats happening here
    ref_predictions = softmax(ref_pred_logits_wo_bias) * (np.expand_dims(np.exp(ref_pred_logcts_wo_bias)[:,0],axis=1)) # final predcitions you can use
    alt_predictions = softmax(alt_pred_logits_wo_bias) * (np.expand_dims(np.exp(alt_pred_logcts_wo_bias)[:,0],axis=1)) # final predcitions you can use
    
    ref_filename = f'{vcf_name}_ref_predictions.npy'
    ref_path = os.path.join(out_path, ref_filename)

    alt_filename = f'{vcf_name}_alt_predictions.npy'
    alt_path = os.path.join(out_path, alt_filename)

    np.save(ref_path, ref_predictions)
    np.save(alt_path, alt_predictions)
    
    # -------------------align predictions---------------------

    ref_al_preds = np.zeros((len(variants), 1000))
    alt_al_preds = np.zeros((len(variants), 1000))

    ref_predictions_sl = ref_predictions[:, np.newaxis, :]
    alt_predictions_sl = alt_predictions[:, np.newaxis, :]

    for i in range(len(variants)):
        var_idx = i
        
        #processing chrombpnet prediction output arrays to be 3 dimensional, since I think that's what supremo lite expects
        
        ref_aligned_1d, alt_aligned_1d = sl.align_predictions_by_coordinate(
            ref_preds=ref_predictions_sl[var_idx],
            alt_preds=alt_predictions_sl[var_idx],
            metadata_row=metadata.iloc[var_idx].to_dict(),
            prediction_type="1D",
            bin_size = 1,
            crop_length = 557
        )

        ref_al_preds[i] = ref_aligned_1d
        alt_al_preds[i] = alt_aligned_1d

    ref_aligned_filename = f'{vcf_name}_ref_predictions_aligned.npy'
    ref_aligned_path = os.path.join(out_path, ref_aligned_filename)

    alt_aligned_filename = f'{vcf_name}_alt_predictions_aligned.npy'
    alt_aligned_path = os.path.join(out_path, alt_aligned_filename)

    np.save(ref_aligned_path, ref_al_preds)
    np.save(alt_aligned_path, alt_al_preds)


    # also get shap scores for counts and profiles


if __name__ == "__main__":
    main()
