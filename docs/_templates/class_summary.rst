.. _{{ fullname }}:

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}

   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}


   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}

   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

