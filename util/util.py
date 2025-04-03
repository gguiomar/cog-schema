import xml.etree.ElementTree as ET

def load_prompt_from_xml(xml, xml_tag, version=None):
    """
    Get the value of a specific XML tag from an XML file.
    
    :param file: The XML file to read.
    :param xml_tag: The XML tag to search for.
    :param version: The version of the prompt. If None, the latest version is used.
    :return: The value of the XML tag.
    """

    root = xml.getroot()

    # Get all <prompt> elements and their versions
    prompts = []
    for prompt in root.findall('prompt'):
        v = int(prompt.attrib.get('version', 0))
        prompts.append((v, prompt))

    if not prompts:
        raise ValueError("No <prompt> elements found in the XML.")

    # Sort by version number and choose the right one
    prompts.sort(reverse=True)  # Highest version first
    if version is None:
        _, selected_prompt = prompts[0]
    else:
        selected_prompt = next((p for v, p in prompts if v == version), None)
        if selected_prompt is None:
            raise ValueError(f"No <prompt> found with version {version}.")

    # Find the tag inside the selected prompt
    tag = selected_prompt.find(xml_tag)
    if tag is None:
        raise ValueError(f"Tag <{xml_tag}> not found in version {version if version is not None else 'latest'}.")

    prompt = tag.text
    prompt = '\n'.join(prompt.splitlines()[1:])
    return prompt
