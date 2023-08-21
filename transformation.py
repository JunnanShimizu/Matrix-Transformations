'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Junnan Shimizu
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''

        self.orig_dataset = orig_dataset
        self.data = data

        pass

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''

        select_data = self.orig_dataset.select_data(headers=headers)

        self.data = data.Data(filepath=None, headers=headers, data=select_data)

        pass

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        pass

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        pass

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        pass

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        pass

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        pass

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''

        ones = np.ones(shape=([np.shape(self.data.data)[0], 1]))
        data1 = np.hstack((self.data.data, ones))

        tdata = np.transpose(data1)

        new_data = C @ tdata

        new_data = np.transpose(new_data)

        new_data = np.delete(new_data, 3, 1)

        self.data = data.Data(headers=self.data.get_headers(), data=new_data, header2col=self.data.get_mappings())
        
        return self.data.data

        pass

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''

        global_min = np.amin(self.data.data)
        global_max = np.amax(self.data.data)
        global_range = global_max - global_min

        # size = np.shape(self.data.data)[1]
        # trans_matrix = np.eye(size + 1)
        # scale_matrix = np.eye(size + 1)

        self.data.data = self.data.data - global_min
        self.data.data = self.data.data / global_range

        # for i in range(5):
        #     trans_matrix[i, i] = global_min
        #     scale_matrix[i, size - 1] = range

        # self.transform(trans_matrix)
        # self.transform(scale_matrix) 
        
        return self.data.data

        pass

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''

        num_cols = int(np.shape(self.data.data)[1])

        for i in range(num_cols):
            local_min = np.amin(self.data.data[:, i])
            local_max = np.amax(self.data.data[:, i])
            data_range = local_max - local_min
            self.data.data[:, i] = self.data.data[:, i] - local_min
            self.data.data[:, i] = self.data.data[:, i] / data_range

        return self.data.data

        pass

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''

        rot_matrix = np.eye(4)
        radians = np.deg2rad(degrees)

        about_axis = self.data.get_index(header)

        if about_axis == 0: # x axis
            rot_matrix[1, 1] = np.cos(radians)
            rot_matrix[1, 2] = -np.sin(radians)
            rot_matrix[2, 1] = np.sin(radians)
            rot_matrix[2, 2] = np.cos(radians)
        elif about_axis == 1: # y axis
            rot_matrix[0, 0] = np.cos(radians)
            rot_matrix[0, 2] = np.sin(radians)
            rot_matrix[2, 0] = -np.sin(radians)
            rot_matrix[2, 2] = np.cos(radians)
        elif about_axis == 2: 
            rot_matrix[0, 0] = np.cos(radians)
            rot_matrix[0, 1] = -np.sin(radians)
            rot_matrix[1, 0] = np.sin(radians)
            rot_matrix[1, 1] = np.cos(radians)
        else:
            print('Out of bounds error')

        return rot_matrix

        pass

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''

        rot_matrix = self.rotation_matrix_3d(header, degrees)
        return self.transform(rot_matrix)

        pass

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''

        plot = plt.scatter(self.orig_dataset.select_data(ind_var), self.orig_dataset.select_data(dep_var), c=self.orig_dataset.select_data(c_var))
        plt.colorbar(plot)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)

        pass

    def scatter(self, ind_var, dep_var, title=None):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''

        plot = plt.scatter(self.data.data[:, 0], self.data.data[:, 1])
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.title(title)

        pass

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''

        fig, axes = plt.subplots(len(data_vars), len(data_vars), figsize=fig_sz, sharex='col', sharey='row')

        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                axes[i, j].scatter(self.data.data[:, j], self.data.data[:, i])

        for i in range(len(data_vars)):
            axes[len(data_vars)-1, i].set_xlabel(data_vars[i])
            axes[i, 0].set_ylabel(data_vars[i])

        fig.suptitle(title)  

        plt.show()    
       
        pass
