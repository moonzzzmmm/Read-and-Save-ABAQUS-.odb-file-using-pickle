"""
    Read the ABAQUS.odb output database into a python object.
    A collection of python built-in libraries is used to construct 
    a virtual object so that users can read and use data in almost 
    the same pattern in the ABAQUS python core environment.

    ABAQUS Version : Abaqus/Explicit 2020
"""

from odbAccess import *
from abaqusConstants import *
from odbMaterial import *
from odbSection import *

import pickle
import numpy as np
import os
from collections import namedtuple


# TODO: Some members/attributes are not involved at the current stage,
#       may be extended in the future.
  
OdbAttr = namedtuple(
    typename = 'OdbAttr',
    field_names = [
        'description',
        'jobData',
        'sections',
        'sectionCategories'
    ]
)

JobData = namedtuple(
    typename = 'JobData',
    field_names = [
        'analysisCode',
        'name', 
        'precision', 
        'version',
        # 'creationTime',
        # 'machineName',
        # 'modificationTime',
        # 'productAddOns', 
    ]
)

# TODO: Other material sections may be extended
HomogeneousShellSection = namedtuple(
    typename = 'HomogeneousShellSection',
    field_names = [
        'name',
        'density',
        'idealization',
        'integrationRule',
        'material',
        'nTemp',
        'numIntPts',
        'poisson',
        'poissonDefinition',
        'preIntegrate',
        'temperature',
        'thickness',
        'thicknessModulus',
        'thicknessType',
        'useDensity',

        # 'nodalThicknessField',
        # 'transverseShear',
        # 'rebarLayers',
        # 'thicknessField',
    ]
)

HomogeneousSolidSection = namedtuple(
    typename = 'HomogeneousSolidSection',
    field_names = [
        'name', 
        'material', 
        'thickness'
    ]
)

RootAssembly = namedtuple(
    typename = 'RootAssembly',
    field_names = [
        # 'connectorOrientations', 
        # 'datumCsyses', 
        # 'pretensionSections', 
        # 'rigidBodies', 
        # 'sectionAssignments',
        'elementSets', 
        'elements', 
        'instances', 
        'name',
        'nodeSets',
        'nodes', 
        'surfaces'
    ]
)

Instance = namedtuple(
    typename = 'Instance',
    field_names = [
        'name',
        'type',
        'embeddedSpace',
        'elements',
        'nodes',
        'elementSets',
        'nodeSets'
    ]
)

# NOTE: Notice that every primitives can be 
#       identified by instance and label,
#       therefore the 'identification' is 
#       a list of:
#        tuple(primitive.instanceName, primitive.label)
PrimitiveSet = namedtuple(
    typename = 'PrimitiveSet',
    field_names = [
        'name',
        'type',
        'elementIdentifier',
        'nodeIdentifier',
        'faces',
        'instanceNames',
        'instances',
        'isInternal',
    ]
)

Element = namedtuple(
    typename = 'Element', 
    field_names = [
        'connectivity',
        'instanceName',
        'instanceNames',
        'label', 
        'sectionCategory',
        'type',
    ]
)

Node = namedtuple(
    typename = 'Node', 
    field_names = [
        'coordinates', 
        'instanceName',
        'label'
    ]
)

SectionPoint = namedtuple(
    typename = 'SectionPoint',
    field_names = [
        'description',
        'number'
    ]
)

SectionCategory = namedtuple(
    typename = 'SectionCategory',
    field_names = [
        'description', 
        'name',
        'sectionPoints'
    ]
)

FieldLocation = namedtuple(
    typename = 'FieldLocation',
    field_names = [
        'position', 
        'sectionPoints'
    ]
)

StepAttr = namedtuple(
    typename = 'StepAttr',
    field_names = [
        'name', 
        'acousticMass',
        'acousticMassCenter',
        'description',
        'domain',
        'inertiaAboutCenter', 
        'inertiaAboutOrigin', 
        'mass', 
        'massCenter',
        'nlgeom', 
        'number', 
        'previousStepName', 
        'procedure', 
        'retainedEigenModes', 
        'timePeriod', 
        'totalTime'
        # 'frames',
        # 'historyRegions',
        # 'loadCases', 
        # 'retainedNodalDofs', 
        # 'eliminatedNodalDofs',
    ]
)

HistoryRegionAttr = namedtuple(
    typename = 'HistoryRegionAttr',
    field_names = [
        'name',
        'description',
        'position'
        # 'loadCase',
        # 'point',
    ]
)

HistoryOutput = namedtuple(
    typename = 'HistoryOutput',
    field_names = [
        # 'conjugateData',
        'name',
        'type',
        'description',
        'data'
    ]
)

FrameAttr = namedtuple(
    typename = 'FrameAttr',
    field_names = [
        'frameId',
        'frameValue',
        'description',
        'domain',
        'incrementNumber',
        'isImaginary',
        # 'fieldOutputs',
        # 'loadCase',
        # 'mode'
        # 'associatedFrame',
        # 'cyclicModeNumber',
        # 'frequency',
    ]
)

FieldOutput = namedtuple(
    typename = 'FieldOutput',
    field_names = [
        'baseElementTypes',
        'componentLabels',
        'description',
        'isComplex',
        'locations',
        'name',
        'type',
        'invariants',
        'bulkDataBlocks',
        # 'validInvariants',
        # 'values',
    ]
)

FieldBulkData = namedtuple(
    typename = 'FieldBulkData',
    field_names = [
        'baseElementType',
        'componentLabels',
        'conjugateData',
        'data',
        'elementLabels',
        'instance',
        'integrationPoints',
        'localCoordSystem',
        'nodeLabels',
        'position',
        'sectionPoint',
        'type'
    ]
)



def foldering(f_dir):
    if not os.path.exists(f_dir):
        os.makedirs(f_dir)

def save_pkl(pkl_file, obj):
    pkl_file = pkl_file.replace("/", "-")
    # NOTE: For some encode-decode considerations,
    #       the open text mode has to be 'wb' and 
    #       the pkl protocol has to be 2
    with open(pkl_file, mode='wb') as f:
        pickle.dump(obj, f, 2)
    del obj

class AbaqusODBPickler:
    
    def __init__(self,
                 odb_file_path, 
                 save_dir='', 
                 odb_read_only=True,
                 odb_read_internal_sets=False
                 ):
        """A convertor for read ABAQUS .odb file using pickle

        Args:
            odb_file_path (str): Path of the .odb file to be read and saved.
            save_dir (str, optional): Root folder directory of the pickled files.
                                      Defaults to empty, and the pickled files will be
                                      saved in the same directory with the odb file.
            odb_read_only (bool, optional): Whether to read only the odb file. Defaults to True.
            odb_read_internal_sets (bool, optional): Whether to read the internal set of the odb file. Defaults to False.
        """
        assert os.path.exists(odb_file_path)
        self.odb_file_path = odb_file_path
        if save_dir:
            self.save_root_dir = save_dir
        else:
            self.save_root_dir = os.path.dirname(odb_file_path)
        self.odb = openOdb(
            path = self.odb_file_path,
            readOnly = odb_read_only,
            readInternalSets = odb_read_internal_sets
        )

    @property
    def root(self):
        return self.odb.rootAssembly    
    
    @property
    def instances(self):
        return self.odb.rootAssembly.instances
    
    @property
    def steps(self):
        return self.odb.steps
    
    @property
    def instance_labels(self):
        return self.odb.rootAssembly.instances.keys()
    
    @property
    def step_labels(self):
        return self.odb.steps.keys()
    
    @property
    def queried_fields(self):
        if isinstance(self._queried_fields, str):
            queried_fields = [self._queried_fields.capitalize()]
        elif hasattr(self._queried_fields, "__iter__"):
            queried_fields = []
            for invariant in self._queried_fields:
                assert isinstance(invariant, str)
                queried_fields.append(invariant.capitalize())
            return queried_fields
        else:
            queried_fields = []
        return queried_fields

    @property
    def queried_invariants(self):
        if isinstance(self._queried_invariants, str):
            queried_invariants = [self._queried_invariants.capitalize()]
        elif hasattr(self._queried_invariants, "__iter__"):
            queried_invariants = []
            for invariant in self._queried_invariants:
                assert isinstance(invariant, str)
                queried_invariants.append(invariant.capitalize())
            return queried_invariants
        else:
            queried_invariants = []
        return queried_invariants

    def to_data(self, ori_data, dtype):
        ''' convert a odb data to numpy data '''
        try:
            if ori_data is None:
                return None
            elif isinstance(ori_data, np.ndarray):
                return ori_data
            else:
                return np.array(ori_data, dtype=dtype)
        except :
            return None
        
    def get_instance(self, instance_label):
        """ get the entire primitives(elements, nodes) from a instance

        Args:
            instance_label (str): label of the instance

        Returns:
            Instance: a virtual collection of the OdbInstance object
        """
        # get OdbInstance object
        instance = self.instances[instance_label]
        # get the list of primitive element data
        elements = [self.get_element(elem) for elem in instance.elements]
        # get the list of primitive node data
        nodes = [self.get_node(node) for node in instance.nodes]

        # get the list of element sets
        element_sets = []
        for elem_set in instance.elementSets.values():
            element_sets.append(self.get_primitive_set(elem_set, 'element'))

        # get the list of node sets
        node_sets = []
        for node_set in instance.nodeSets.values():
            node_sets.append(self.get_primitive_set(node_set, 'node'))

        instance = Instance(
                name = str(instance.name),
                type = str(instance.type),
                embeddedSpace = str(instance.embeddedSpace),
                elements = elements,
                nodes = nodes,
                elementSets = element_sets,
                nodeSets = node_sets
            )
        return instance

    def get_primitive_set(self, primitive_set, set_type):
        ''' convert a OdbSet of primitives (nodes, elements) '''
        
        element_identifiers = []
        node_identifiers = []
        elements = primitive_set.elements
        nodes = primitive_set.nodes
        # NOTE: For primitiveSets of a rootAssembly, the primitives are contained in a tuple.
        if isinstance(elements, tuple):
            elements = elements[0]
        if isinstance(nodes, tuple):
            nodes = nodes[0]
        # NOTE: The base information of all the primitives are obtained already. 
        #       Thus, only labels are collected for identification.
        if nodes:
            for node in nodes:
                node_identifiers.append((node.instanceName, int(node.label)))
        if elements:
            for element in elements:
                element_identifiers.append((element.instanceName, int(element.label)))
        primitive_set_collection = PrimitiveSet(
            name = str(primitive_set.name),
            type = str(set_type),
            elementIdentifier = element_identifiers,
            nodeIdentifier = node_identifiers,
            faces = str(primitive_set.faces),
            instanceNames = str(primitive_set.instanceNames),
            instances = str(primitive_set.instances),
            isInternal = str(primitive_set.isInternal),
        )
        return primitive_set_collection

    def get_section_points(self, section_points):
        ''' convert a sectionPoints to a list '''

        section_point_list = []
        if section_points:
            for section_point in section_points:
                section_point_list.append(
                    SectionPoint(
                        description = str(section_point.description),
                        number = int(section_point.number)
                    )
                )
        return section_point_list
    
    def get_element(self, element):
        ''' convert a OdbMeshElement to a list '''

        # NOTE: The sectionCategory of a element can be queried 
        #       in odb.sectionCategorys collection
        element_collection = Element(
            connectivity = list(element.connectivity), 
            instanceName = str(element.instanceName), 
            instanceNames = list(element.instanceNames), 
            label = int(element.label), 
            sectionCategory = element.sectionCategory.name, 
            type = str(element.type)
        )
        return element_collection
    
    def get_node(self, node):
        ''' convert a OdbMeshNode to a list '''

        node_collection = Node(
            coordinates = list(node.coordinates),
            instanceName = str(node.instanceName),
            label = int(node.label)
        )
        return node_collection
    
    def get_model_data(self):
        """convert the model data start from rootAssembly

        Returns:
            RootAssembly: a collection of the model database
        """

        root_dir = self.save_root_dir + "\\rootAssembly"
        foldering(root_dir)
        
        instance_dir = root_dir + "\\instances"
        foldering(instance_dir)
        
        element_sets_dir = root_dir + "\\elementSets"
        foldering(element_sets_dir)
        
        node_sets_dir = root_dir + "\\nodeSets"
        foldering(node_sets_dir)
        
        surfaces_dir = root_dir + "\\surfaces"
        foldering(surfaces_dir)

        for instance_label in self.instance_labels:
            instance = self.get_instance(instance_label)
            save_pkl(instance_dir + "\\" + instance.name + ".pkl", instance)
        
        # get the list of root element data
        root_elements = [self.get_element(elem) for elem in self.root.elements]
        save_pkl(root_dir + "\\elements.pkl", root_elements)
        
        # get the list of root node data
        root_nodes = [self.get_node(node) for node in self.root.nodes]
        save_pkl(root_dir + "\\nodes.pkl", root_nodes)
        
        # get the list of root element sets
        for elem_set in self.root.elementSets.values():
            root_element_set = self.get_primitive_set(elem_set, 'element')
            save_pkl(element_sets_dir + "\\" + root_element_set.name + ".pkl", root_element_set)

        # get the list of root node sets
        for node_set in self.root.nodeSets.values():
            root_node_set = self.get_primitive_set(node_set, 'node')
            save_pkl(node_sets_dir + "\\" + root_node_set.name + ".pkl", root_node_set)
        
        # get the list of root surface sets
        for surface_set in self.root.surfaces.values():
            root_surface = self.get_primitive_set(surface_set, 'surface')
            save_pkl(surfaces_dir + "\\" + root_surface.name + ".pkl", root_surface)
        
    def get_history_outputs(self, history_outputs, history_region_dir):
        ''' convert a historyOutputs object data '''

        history_outputs_root_dir = history_region_dir + "\\historyOutputs"
        foldering(history_outputs_root_dir)
        for hist_output_label in history_outputs.keys():
            history_output = history_outputs[hist_output_label]

            history_output_data = np.array(history_output.data, dtype=np.float64)
            history_output_collection = HistoryOutput(
                name = str(history_output.name),
                type = str(history_output.type.getText()),
                description = str(history_output.description),
                data = history_output_data
            )
            save_pkl(history_outputs_root_dir + "\\" + hist_output_label + ".pkl", history_output_collection)

    def get_field_bulk_data(self, field_bulk_data_blocks):
        """convert FieldBulkData to a collection
            NOTE: The reason for not getting data from fieldOutput.values is the 
                  FieldBulkData objects are built in a integrated NDAarray object,
                  which avoids reading potentially duplicate data blocks with a loop.
                  This can effectively read the entire field data, saving disk storage space.

        Args:
            field_bulk_data_blocks (List[FieldBulkData]): The FieldBulkData object represents 
            the entire field data for a class of elements or nodes. 
            All elements in a class correspond to the same element type and material.

        Returns:
            field_bulk_data_collections: List[FieldBulkData]
        """

        field_bulk_data_collections = []
        for bulk_data_block in field_bulk_data_blocks:
            if bulk_data_block.sectionPoint is not None:
                field_bulk_data_section_point = SectionPoint(
                    description = bulk_data_block.sectionPoint.description, 
                    number = bulk_data_block.sectionPoint.number),
            else:
                field_bulk_data_section_point = None
            field_bulk_data_collection = FieldBulkData(
                baseElementType = str(bulk_data_block.baseElementType),
                componentLabels = str(bulk_data_block.componentLabels),
                conjugateData = self.to_data(bulk_data_block.conjugateData, np.float64),
                data = self.to_data(bulk_data_block.data, np.float64),
                elementLabels = self.to_data(bulk_data_block.elementLabels, np.int32),
                nodeLabels = self.to_data(bulk_data_block.nodeLabels, np.int32),
                instance = str(bulk_data_block.instance.name),
                integrationPoints = self.to_data(bulk_data_block.integrationPoints, np.int32),
                localCoordSystem = self.to_data(bulk_data_block.localCoordSystem, np.float64),
                position = str(bulk_data_block.position.getText()),
                sectionPoint = field_bulk_data_section_point,
                type = bulk_data_block.type.getText()
            )
            field_bulk_data_collections.append(field_bulk_data_collection)
        return field_bulk_data_collections

    def get_field(self, field):
        ''' convert FieldOutput data to collection '''

        # get field data
        # NOTE: Only the original data will be read and save by default,
        #       for the storage and efficiency consideration.
        #       Every invariants (including Mises and Magnitude) should be required
        #       by configuring queried_invariants.
        bulk_data_collecion = self.get_field_bulk_data(field.bulkDataBlocks)

        # get invariant fields
        invariant_bulk_data_collecions = {}
        field_valid_invariants = [invariant_obj for invariant_obj in field.validInvariants]
        field_valid_invariants_text = [invariant.getText() for invariant in field.validInvariants]
        if self.queried_invariants == 'ALL':
            field_queried_invariants = field_valid_invariants_text
        else:
            field_queried_invariants = self.queried_invariants
        for invariant in field_queried_invariants:
            if invariant in field_valid_invariants_text:
                invariant_abq_obj = field_valid_invariants[field_valid_invariants_text.index(invariant)]
                invariant_field = field.getScalarField(invariant_abq_obj)
                invariant_bulk_data_collecion = self.get_field_bulk_data(invariant_field.bulkDataBlocks)
                invariant_bulk_data_collecions[invariant_field.name] = invariant_bulk_data_collecion

        # get field location data list
        field_location_list = []
        for loc in field.locations:
            section_point_list = self.get_section_points(loc.sectionPoints)
            location_collection = FieldLocation(
                position = str(loc.position.getText()),
                sectionPoints = section_point_list
            )
            field_location_list.append(location_collection)

        field_component_labels = [str(label) for label in field.componentLabels]
        field_collection = FieldOutput(
            name = str(field.name),
            baseElementTypes = list(field.baseElementTypes),
            type = str(field.type.getText()),
            description = str(field.description),
            isComplex = str(field.isComplex.getText()),
            componentLabels = field_component_labels,
            invariants = invariant_bulk_data_collecions,
            locations = field_location_list,
            bulkDataBlocks = bulk_data_collecion
        )
        return field_collection

    def get_section_category(self, section_category):
        """convert ABAQUS SectionCategory to a collection

        Args:
            section_category (SectionCategory): ABAQUS SectionCategory object
        """

        section_point_list = self.get_section_points(section_category.sectionPoints)
        section_category_collection = SectionCategory(
            description = str(section_category.description),
            name = str(section_category.name),
            sectionPoints = section_point_list
        )
        return section_category_collection

    def get_frame(self, frame, frame_dir):
        ''' convert Frame data to collections '''

        field_outputs_root_dir = frame_dir + "\\fieldOutputs"
        foldering(field_outputs_root_dir)
        field_outputs = frame.fieldOutputs
        if self.queried_fields == 'ALL':
            queried_fields = [field_label for field_label in field_outputs.keys()]
        else:
            queried_fields = self.queried_fields
        for field_label in field_outputs.keys():
            if field_label.capitalize() in queried_fields:
                field = field_outputs[field_label]
                field_collection = self.get_field(field)
                save_pkl(field_outputs_root_dir + "\\" + field_label + ".pkl", field_collection)

        frame_attr_collection = FrameAttr(
            frameId = int(frame.frameId),
            frameValue = float(frame.frameValue),
            description = str(frame.description),
            domain = str(frame.domain.getText()),
            incrementNumber = int(frame.incrementNumber),
            isImaginary = bool(frame.isImaginary),
        )
        save_pkl(frame_dir + "\\attr.pkl", frame_attr_collection)

    def get_results_data(self, st_frame, ed_frame):
        ''' convert OdbStep data to collections '''

        steps_root_dir = self.save_root_dir + "\\steps"
        foldering(steps_root_dir)
        
        for step_label in self.step_labels:
            step = self.steps[step_label]
            step_dir = steps_root_dir + "\\" + step.name
            foldering(step_dir)
            
            # get histrory outputs from history regions
            history_regions_root_dir = step_dir + "\\historyRegions"
            foldering(history_regions_root_dir)
            history_regions = step.historyRegions
            for region_label in history_regions.keys():
                history_region = history_regions[region_label]
                history_region_dir = history_regions_root_dir + "\\" + history_region.name
                foldering(history_region_dir)

                # get history outputs
                self.get_history_outputs(history_region.historyOutputs, history_region_dir)
                history_region_attr = HistoryRegionAttr(
                    name = str(history_region.name),
                    description = str(history_region.description),
                    position = str(history_region.position.getText()),
                )
                save_pkl(history_region_dir + "\\attr.pkl", history_region_attr)
            
            # get field outputs from frames
            frames_root_dir = step_dir + "\\frames"
            foldering(frames_root_dir)
            frames = step.frames
            
            n_frames = len(frames)
            if ed_frame == -1: ed_frame = n_frames
            assert 1 <= st_frame <= ed_frame <= n_frames
            for f_idx, frame in enumerate(frames):
                if (f_idx >= st_frame) and (f_idx <= ed_frame):
                    frame_dir = frames_root_dir + "\\" + str(f_idx)
                    foldering(frame_dir)
                    self.get_frame(frame, frame_dir)
            
            step_attr_collection = StepAttr(
                name = str(step.name),
                acousticMass = float(step.acousticMass),
                acousticMassCenter = list(step.acousticMassCenter),
                description = str(step.description),
                domain = str(step.domain.getText()),
                inertiaAboutCenter = list(step.inertiaAboutCenter),
                inertiaAboutOrigin = list(step.inertiaAboutOrigin),
                mass = float(step.mass),
                massCenter = list(step.massCenter),
                nlgeom = bool(step.nlgeom),
                number = int(step.number),
                previousStepName = str(step.previousStepName),
                procedure = str(step.procedure),
                retainedEigenModes = list(step.retainedEigenModes),
                timePeriod = float(step.timePeriod),
                totalTime = float(step.totalTime),
            )
            save_pkl(step_dir + "\\attr.pkl", step_attr_collection)

    def get_section(self, section):
        """convert a Section object to a collection

        Args:
            section (*Section): ABAQUS material section object
        """

        section_class_label = str(section.__class__)
        if  section_class_label == "<type 'HomogeneousShellSection'>":
            section_collection = HomogeneousShellSection(
                name = section.name,
                density = float(section.density),
                idealization = section.idealization.getText(),
                integrationRule = section.integrationRule.getText(),
                material = section.material,
                nTemp = int(section.nTemp),
                numIntPts = int(section.numIntPts),
                poisson = float(section.poisson),
                poissonDefinition = section.poissonDefinition.getText(),
                preIntegrate = bool(section.preIntegrate),
                temperature = section.temperature.getText(),
                thickness = float(section.thickness),
                thicknessModulus = self.to_data(section.thicknessModulus, np.float32),
                thicknessType = section.thicknessType.getText(),
                useDensity = bool(section.useDensity),
            )
        elif section_class_label == "<type 'HomogeneousSolidSection'>":
            section_collection = HomogeneousSolidSection(
                name = section.name,
                material = section.material,
                thickness = section.thickness
            )
        else:
            section_collection = "UNRESOLVED SECTION"
        return section_collection

    def get_odb(self):
        """convert a Abaqus Odb object to a collection
        """

        job_data_collection = JobData(
            name = self.odb.jobData.name,
            analysisCode = self.odb.jobData.analysisCode.getText(),
            precision = self.odb.jobData.precision.getText(),
            version = self.odb.jobData.version,
        )

        section_collections = {}
        for section_name in self.odb.sections.keys():
            section = self.odb.sections[section_name]
            section_collection = self.get_section(section)
            section_collections[section_name] = section_collection

        section_category_collections = {}
        for section_category_name in self.odb.sectionCategories.keys():
            section_category = self.odb.sectionCategories[section_category_name]
            section_category_collection = self.get_section_category(section_category)
            section_category_collections[section_category_name] = section_category_collection

        odb_collection = OdbAttr(
            description = self.odb.description,
            jobData = job_data_collection,
            sections = section_collections,
            sectionCategories = section_category_collections,
        )
        save_pkl(self.save_root_dir + "\\attr.pkl", odb_collection)

    def _configure_queried_fields(self, queried_fields, queried_invariants):
        self._queried_fields = queried_fields
        self._queried_invariants = queried_invariants
        
    def struct(self,
               read_odb_obj=True,
               read_model_data=True,
               st_frame=1, 
               ed_frame=-1, 
               queried_fields='ALL',
               queried_invariants=['MISES', 'MAGNITUDE']):
        
        """read and save odb data to the directory of self.save_root_dir
        
        Args:
            read_odb_obj (bool, optional): Whether to read the attributes data of the ODB object. Defaults to True.
            read_model_data (bool, optional): Whether to read the model data of the odb.rootAssembly object. Defaults to True.
            st_frame (int, optional): The starting frame. Defaults to 1.
            ed_frame (int, optional): The ending frame. Defaults to -1.
            queried_fields (str|sequence[str], optional): The fieldOutput labels to read. Defaults to 'ALL'.
            queried_invariants (list, optional): The invariant labels of a fieldOutput to read. Defaults to ['MISES', 'MAGNITUDE'].
        """
        assert isinstance(st_frame, int), isinstance(ed_frame, int)
        self._configure_queried_fields(queried_fields, queried_invariants)
        foldering(self.save_root_dir)
        if read_odb_obj:
            self.get_odb()
        if read_model_data:
            self.get_model_data()
        self.get_results_data(st_frame, ed_frame)
        print('>>>> Read and Save Done')

    def close(self):
        """ close the odb file """
        self.odb.close()

if __name__ == "__main__":
    abq_pickler = AbaqusODBPickler(
        odb_file_path=r"TEMP\Job-025.odb",
        save_dir=r"Job-025"
    )
    abq_pickler.struct(
        queried_fields = ['UT', 'PEEQ', 'A'],
        queried_invariants = ['MISES'],
        st_frame = 10,
        ed_frame = 30
    )

