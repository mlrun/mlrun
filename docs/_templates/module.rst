{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:
   :undoc-members:

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree: _autosummary
   :template: module.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}