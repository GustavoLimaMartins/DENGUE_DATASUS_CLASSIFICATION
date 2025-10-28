from data_tools_analisys.f_modeling_setup_and_evaluate import ModelingTools, ModelType
import models.g_model_ml as machine_learning
import models.g_model_dl as deep_learning
from data_tools_analisys.d_data_analisys import analisarDadosViaGraficos

analisarDadosViaGraficos()
tools = ModelingTools()
types = ModelType()
machine_learning.main(tools, types)
deep_learning.main(tools)
