import blimpy
import h5py
import sys



if __name__ == "__main__":
    # Read the existing .h5 file using blimpy
    file_path = sys.argv[1]
    data = blimpy.Waterfall(file_path)

    # Access the data and metadata as needed
    header = data.header
    spectrogram = data.data
    print(data.shape)

    # Create a new .h5 file using h5py
    output_file_path = 'path/to/your/output_file.h5'
    with h5py.File(output_file_path, 'w') as output_file:
        # Create datasets in the new file
        output_file.create_dataset('header', data=header)
        output_file.create_dataset('spectrogram', data=spectrogram)

    print("New .h5 file created successfully!")
