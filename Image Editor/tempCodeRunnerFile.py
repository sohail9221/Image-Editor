     # Define output image
        output = np.zeros_like(image)
        # Get kernel size and center
        k_size = kernel.shape[0]
        k_center = k_size // 2
        
        # Iterate over the image
        for i in range(k_center, image.shape[0] - k_center):
            for j in range(k_center, image.shape[1] - k_center):
                # Apply erosion operation
                min_val = 255
                for ki in range(k_size):
                    for kj in range(k_size):
                        if kernel[ki][kj] == 1:
                            val = image[i + ki - k_center][j + kj - k_center]
                            min_val = min(min_val, val)
                output[i][j] = min_val
        
        return output