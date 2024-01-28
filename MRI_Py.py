# Importing required modules
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit
import os
import nibabel
import numpy as np
from sklearn.cluster import KMeans as km
import nighres
import levelsetplotter as lsp

# Designating all file-output locations
# out_dir to be set by user
out_dir = ''
skullstripdir = out_dir + '/skullstrip/'
skullstripintdir = out_dir + '/skullstripint/'
mgdmdir = out_dir + '/mgdm_results/'
bexdir = out_dir + '/brain_extraction/'
levelsets = out_dir + '/levelsets/'
meshes = out_dir + '/meshes/'
cruise = out_dir + '/cruise/'

# Designating patient brains that will be used in cortical analysis located in /data/projects/ahead/raw_gdata/PATIENT
root='/data/projects/ahead/raw_gdata/'

Patient = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]

# Performing Nighres segmentation on patient brains in list "Patient"
try:
    for i in range(0, len(Patient)):
        print('processing patient:' + Patient[i])

        skullstripping_results = nighres.brain.mp2rage_skullstripping(

            second_inversion=out_dir + Patient[i] + '/nii/ReOriented/inv2_te1_m_corr_reoriented.nii.gz',
            t1_weighted=out_dir + Patient[i] + '/nii/ReOriented/mt1corr_reoriented.nii.gz',
            t1_map=out_dir + Patient[i] + '/nii/ReOriented/inv1_te1_m_corr_reoriented.nii.gz',
            save_data=True,
            file_name=Patient[i],
            output_dir=skullstripdir
        )
        mgdm_results = nighres.brain.mgdm_segmentation(
            contrast_image1=skullstripping_results['t1w_masked'],
            contrast_type1="Mp2rage7T",
            contrast_image2=skullstripping_results['t1map_masked'],
            contrast_type2="T1map7T",
           # contrast_image1=out_dir + Patient[i] + '/nii/mt1corr.nii.gz',
           # contrast_type1='T1map7T',
            #contrast_image2=out_dir + Patient[i] + '/nii/t1corr.nii.gz',
            #contrast_type2='Mp2rage7T',
            n_steps=5,
            topology='wcs',
            adjust_intensity_priors=True,
            normalize_qmaps=False,
            save_data=True,
            file_name=Patient[i],
            output_dir=mgdmdir
        )
            # For loop for extracting both cerebra and their respective cortices
        for b in range(0, 2):
            ereg = ['left_cerebrum', 'right_cerebrum']
            bex = nighres.brain.extract_brain_region(
                segmentation=mgdmdir + Patient[i] + '_mgdm-seg.nii.gz',
                levelset_boundary=mgdmdir + Patient[i] + '_mgdm-dist.nii.gz',
                maximum_membership=mgdmdir + Patient[i] + '_mgdm-mems.nii.gz',
                maximum_label=mgdmdir + Patient[i] + '_mgdm-lbls.nii.gz',
                extracted_region=ereg[b],
                save_data=True,
                output_dir=bexdir + '/' + ereg[b]
            )
            cruisefunc = nighres.cortex.cruise_cortex_extraction(
                init_image=bex['inside_mask'],
                wm_image=bex['inside_proba'],
                gm_image=bex['region_proba'],
                csf_image=bex['background_proba'],
                normalize_probabilities=True,
                save_data=True,
                output_dir=cruise + '/' + ereg[b]
            )
            print('Extracting intensities 0-3 from gwb nifti.')
            gwb = nibabel.load(cruisefunc['gwb'])
            intensities = gwb.get_fdata()
            zerothree = ((intensities > -3) * (intensities < 3)) > 0


            print('Saving zerothree img')
            zerothree_img = nibabel.Nifti1Image(zerothree, gwb.affine, gwb.header)
            zerothree_img_inv = nibabel.Nifti1Image(~zerothree, gwb.affine, gwb.header)
            nibabel.save(zerothree_img, cruise + '/' + ereg[b] + '/zerothree.nii.gz')
            nibabel.save(zerothree_img_inv, cruise + '/' + ereg[b] + '/zerothreeinv.nii.gz')
            print('Done')

            print('Starting K-means Clustering on zerothree images')
            zerothreenifti = nibabel.load(cruise + '/' + ereg[b] + '/zerothree.nii.gz')
            zerothreefdata = zerothreenifti.get_fdata()
            kmeansmask = np.where(zerothreefdata > 0, intensities, zerothreefdata)
            kmeansnifti = nibabel.Nifti1Image(kmeansmask, gwb.affine, gwb.header)
            nibabel.save(kmeansnifti, out_dir + f"{Patient[i]}kmeansnifti.nii.gz")
            ztkmeans = kmeansnifti.get_fdata()
            coord_list = np.argwhere(zerothreefdata == 1)
            print('Starting alghorithm')
            #if not os.path.exists(cruise + ereg[b] + '/' + Patient[i] + '800_-3_3_kmeans.nii.gz'):
            # KMeans clustering parameters initialization
            to_kmeans = km(
                # Method for initialization, default is k-means++, other option is 'random', learn more at scikit-learn.org
                init='k-means++',
                # Number of clusters to be generated, int, default=8
                n_clusters=800,
                # n_init is the number of times the k-means algorithm will be ran with different centroid seeds, int, default=10
                n_init=1,
                # maximum iterations, int, default=300
                max_iter=1,
                # relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive
                # iterations to declare convergence, float, default=0.0001
                tol=0.00005,
                # verbosity, int, default=0
                verbose=1,
                # random state instance or None, default=None
                random_state=None,
                # copy_x, bool, default=True, makes sure original data is not overwritten
                copy_x=True,
                # algorithm, dict, {lloyd, elkan}, default="lloyd", classic EM-style algorithm is lloyd, elkan can be more efficient on datasets with well-defined clusters
                algorithm='lloyd')
            # Performing KMeans clustering on the voxels with an intensity of 1
            kmeans = to_kmeans.fit(coord_list)
            # Extracting labels of clustered datapoints
            labels = to_kmeans.labels_
            print('Highest label found: ', np.max(labels))
            # Setting up an empty array with the shape of the input nifti file
            clustered_data = np.zeros(zerothreefdata.shape, dtype=np.uint16)
            red_green_fit = np.zeros(zerothreefdata.shape, dtype=np.uint16)
            intensity_fit = np.zeros(zerothreefdata.shape, dtype=np.uint16)
            # Stepwise method for transforming the flattened input array back into 3d space
            for l in range(len(coord_list)):
                x, y, z = coord_list[l]
                # Overwriting data with labels, adding 1 so no labels have the same intensity as the background
                clustered_data[x, y, z] = labels[l] + 1
            #for l in range(len(coord_list)):
            #    x, y, z = coord_list[l]
            #    # Overwriting data with goodbool binary value to specify if fit was of sufficient quality
            #    clustered_data[x, y, z] = goodbool[l]
            # Saving clustered image
            kmeans_img = nibabel.Nifti1Image(clustered_data, kmeansnifti.affine, kmeansnifti.header)
            print('Saving kmeans_img as' + cruise + ereg[b] + '/' + Patient[i] + 'kmeans.nii.gz')
            nibabel.save(kmeans_img, cruise + ereg[b] + '/' + Patient[i] + '800_-3_3_kmeans.nii.gz')
            print('done')


            def sigmoidal(x, L, x0, k, b):
                y = L / (1 + np.exp(-k * (x - x0))) + b
                return y

            def load_nifti_data(filename):
                img = nibabel.load(filename)
                data = img.get_fdata()
                return data

            def plot_cluster_data(cluster_data, levelset_data, t1_data, cluster_index, cluster_filename):
                current_cluster = cluster_index.astype(numpy.int64)
                print(f'Current cluster: {current_cluster}')
                cluster_mask = (cluster_data == cluster_index)
                levelset_values = levelset_data[cluster_mask]
                t1_values = t1_data[cluster_mask]
                plt.scatter(levelset_values, t1_values, label=f'Cluster {cluster_index}')
                plt.xlabel('Levelset Value')
                plt.ylabel('T1 Intensity')
                plt.legend()
                cluster_location = np.argwhere(cluster_data == current_cluster+1)
                print('Current cluster location:')
                print(cluster_location)

                try:
                    # Fit the data to the sigmoidal function
                    p0 = [max(t1_values), np.median(levelset_values), 1, min(t1_values)]
                    popt, pcov = curve_fit(f=sigmoidal, xdata=levelset_values, ydata=t1_values, p0=p0, maxfev=5000, method="lm")
                    # Generate the fitted curve
                    fitted_t1_values = sigmoidal(np.sort(levelset_values), *popt)
                    avg_par_variance = np.sum(np.diag(pcov)) / len(np.diag(pcov))
                    if avg_par_variance > 0.01:
                        # Plot the fitted curve in red
                        plt.plot(np.sort(levelset_values), fitted_t1_values, 'r-', label='Fitted Curve')
                        plot_filename = f"{cluster_filename}_cluster_{cluster_index}_cov_{avg_par_variance}.png"
                        plt.savefig(f"cluster_plots/bad_{plot_filename}")
                        plt.close()
                        # Overwriting data with 1, signifying a cluster with a bad fit
                        for r in range(len(cluster_location)):
                            xc, yc, zc = cluster_location[r]
                            # Overwriting data with labels, adding 1 so no labels have the same intensity as the background
                            red_green_fit[xc, yc, zc] = 1
                            intensity_fit[xc, yc, zc] = 1/avg_par_variance
                    else:
                        # Plot the fitted curve in green
                        plt.plot(np.sort(levelset_values), fitted_t1_values, 'g-', label='Fitted Curve')
                        plot_filename = f"{cluster_filename}_cluster_{cluster_index}_cov_{avg_par_variance}.png"
                        plt.savefig(f"cluster_plots/good_{plot_filename}")
                        plt.close()
                        # Overwriting data with 2, signifying a cluster with a good fit
                        for r in range(len(cluster_location)):
                            xc, yc, zc = cluster_location[r]
                            # Overwriting data with labels, adding 1 so no labels have the same intensity as the background
                            red_green_fit[xc, yc, zc] = 2
                            intensity_fit[xc, yc, zc] = 1 / avg_par_variance
                        # Print a status bar to the console
                        print(f"Cluster {cluster_index} plotted and saved as {plot_filename}")
                except Exception as e:
                    print(f"Error fitting cluster {cluster_index}: {str(e)}")

            def main(cluster_file, levelset_file, t1_file):
                # Load NIfTI files
                cluster_data = load_nifti_data(cluster_file)
                levelset_data = load_nifti_data(levelset_file)
                t1_data = load_nifti_data(t1_file)

                # Determine the unique cluster values
                unique_clusters = np.unique(cluster_data)

                # Create a directory to save the plots
                output_directory = "cluster_plots"
                os.makedirs(output_directory, exist_ok=True)

                for cluster_index in unique_clusters:
                    #if cluster_index == 0:  # Skip background cluster if present
                    #    continue

                    # Specify the cluster filename without extension for saving plots
                    cluster_filename = os.path.splitext(os.path.basename(cluster_file))[0]

                    plot_cluster_data(cluster_data, levelset_data, t1_data, cluster_index, cluster_filename)

                red_green_image = nibabel.Nifti1Image(red_green_fit, kmeansnifti.affine, kmeansnifti.header)
                nibabel.save(red_green_image, cruise + ereg[b] + '/' + Patient[i] + 'red_green_800_-3_3_kmeans.nii.gz')
                intensity_image = nibabel.Nifti1Image(intensity_fit, kmeansnifti.affine, kmeansnifti.header)
                nibabel.save(intensity_image, cruise + ereg[b] + '/' + Patient[i] + 'intensity_800_-3_3_kmeans.nii.gz')

            if __name__ == "__main__":
                cluster_file = cruise + ereg[b] + '/' + Patient[i] + '800_-3_3_kmeans.nii.gz'
                if os.path.exists(cruise + ereg[b] + '/' + Patient[i] + '_mgdm-seg_xproba-rcrgm_cruise-gwb.nii.gz'):
                    levelset_file = cruise + ereg[b] + '/' + Patient[i] + '_mgdm-seg_xproba-rcrgm_cruise-gwb.nii.gz'
                else:
                    levelset_file = cruise + ereg[b] + '/' + Patient[i] + '_mgdm-seg_xproba-lcrgm_cruise-gwb.nii.gz'
                t1_file = '/data/projects/ahead/raw_gdata/' + Patient[i] + '/nii/ReOriented/r1corr_reoriented.nii.gz'

                main(cluster_file, levelset_file, t1_file)
except Exception as e2:
    print(f"Error occurred: {str(e2)}")
