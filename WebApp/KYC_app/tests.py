from django.test import TestCase

# Create your tests here.
# Example standalone testing code
image_data = "data:image/png;base64,<your_base64_string_here>"
is_live, message = process_image(image_data)
print(is_live, message)
